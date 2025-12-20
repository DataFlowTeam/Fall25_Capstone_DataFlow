# realtime_decode.py
# Realtime ASR + Punctuation + SAFE ITN + CIF (proper integrate-and-fire -> c_i -> CTC)
#
# CIF here is implemented "chuẩn bài" theo cơ chế:
#   encoder H_u -> alpha_u (Conv1D + FC + sigmoid) -> integrate-and-fire -> c_i
#   rồi chạy CTC trên chuỗi c_i (không dùng decoder).
#
# Assumption: bạn đã train CIF head tương ứng và lưu checkpoint là cif_best.pt
# (chỉ head alpha / hoặc full head conv+fc). Script sẽ load và chạy inference.

import os
import math
import argparse
import yaml
import re
import sys
import threading
import queue
import contextlib
from collections import deque
import traceback

from api.private_config import *  # CHUNKFORMER_CHECKPOINT, ITN_REPO, ...

import numpy as np
import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as kaldi
import sounddevice as sd
import onnxruntime as ort
from punctuators.models import PunctCapSegModelONNX
import colorama

import pynini
from pynini.lib.rewrite import top_rewrite
from pynini.lib import rewrite

# ---------- Bootstrap import path (EduAssist as Sources Root) ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.services.chunkformer.model.utils.init_model import init_model
from api.services.chunkformer.model.utils.checkpoint import load_checkpoint
from api.services.chunkformer.model.utils.file_utils import read_symbol_table
from api.services.chunkformer.model.utils.ctc_utils import get_output

# ----------------------------------------------------------------------

# ANSI
colorama.init(autoreset=True)
YELLOW = colorama.Fore.YELLOW
BLUE   = colorama.Fore.BLUE

# ===== Regex & helpers =====
WORD_RE       = re.compile(r"[0-9A-Za-zÀ-Ỵà-ỵ]+")
_TOKEN_RE     = re.compile(r"\S+")
END_SENT_RE   = re.compile(r"[\.!\?…]\s*$", re.UNICODE)
SENT_PUNCT_RE = re.compile(r"[\.!\?…]")

NUM_TOKEN_RE = re.compile(r"^[0-9]+([.,][0-9]+)*$")

VI_NUM_WORDS = {
    "không", "một", "mốt", "hai", "ba", "bốn", "tư",
    "năm", "lăm", "sáu", "bảy", "tám", "chín",
    "mười", "mươi", "trăm", "nghìn", "ngàn", "triệu",
    "tỷ", "tỉ", "linh", "lẻ", "phẩy"
}

PUNCT_CHARS = ".,;:!?…"


def strip_punct(tok: str) -> str:
    return tok.strip(PUNCT_CHARS)


def is_digit_token(tok: str) -> bool:
    t = strip_punct(tok)
    return bool(NUM_TOKEN_RE.match(t))


def is_vi_num_word(tok: str) -> bool:
    t = strip_punct(tok).lower()
    return t in VI_NUM_WORDS


def has_mixed_num_span(text: str) -> bool:
    tokens = _TOKEN_RE.findall(text)
    spans = []
    cur = []
    for tok in tokens:
        if is_digit_token(tok) or is_vi_num_word(tok):
            cur.append(tok)
        else:
            if cur:
                spans.append(cur)
                cur = []
    if cur:
        spans.append(cur)

    for span in spans:
        has_digits = any(is_digit_token(t) for t in span)
        has_words  = any(is_vi_num_word(t) for t in span)
        if has_digits and has_words:
            return True
    return False


def advance_pointer_by_words(full_text: str, start_idx: int, n_words: int) -> int:
    cnt = 0
    for m in WORD_RE.finditer(full_text, start_idx):
        cnt += 1
        if cnt == n_words:
            return m.end()
    return len(full_text)


def longest_suffix_prefix_overlap(a: str, b: str, max_k: int = 32) -> int:
    k = min(max_k, len(a), len(b))
    for L in range(k, 0, -1):
        if a.endswith(b[:L]):
            return L
    return 0


def split_commit_region(punct_text: str, window_words: int, margin_words: int):
    punct_text = punct_text.strip()
    if not punct_text:
        return "", ""

    tokens = WORD_RE.findall(punct_text)
    if len(tokens) <= window_words:
        return "", punct_text

    boundary_idx = None
    for m in SENT_PUNCT_RE.finditer(punct_text):
        idx = m.end()
        remaining_words = len(WORD_RE.findall(punct_text[idx:]))
        if remaining_words >= margin_words:
            boundary_idx = idx

    if boundary_idx is None:
        return "", punct_text

    commit_prefix = punct_text[:boundary_idx]
    active_text   = punct_text[boundary_idx:]
    return commit_prefix, active_text


# =========================
# CIF (proper I&F -> c_i)
# =========================
class CIFAlphaHead(nn.Module):
    """
    Alpha head đúng theo slide:
      H (B,T,C) -> Conv1D (time) -> FC -> sigmoid => alpha (B,T) in (0,1)

    Lưu ý: đây chỉ là head dự đoán alpha; cơ chế integrate-and-fire tạo c_i nằm ở code.
    """
    def __init__(self, in_dim: int, conv_out_dim: int = None, kernel_size: int = 3):
        super().__init__()
        if conv_out_dim is None:
            conv_out_dim = in_dim
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_dim, conv_out_dim, kernel_size=kernel_size, padding=pad, bias=True)
        self.fc   = nn.Linear(conv_out_dim, 1, bias=True)

    def forward(self, H_btc: torch.Tensor) -> torch.Tensor:
        # H_btc: (B,T,C)
        x = H_btc.transpose(1, 2)     # (B,C,T)
        x = self.conv(x)              # (B,C',T)
        x = x.transpose(1, 2)         # (B,T,C')
        z = self.fc(x).squeeze(-1)    # (B,T)
        return torch.sigmoid(z)       # (B,T)


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


@torch.no_grad()
def init_cif_alpha_head(args, encoder_out_dim: int):
    """
    Load alpha head từ cif_best.pt.
    Script cố gắng infer kernel/dim từ conv.weight nếu có.
    """
    ckpt_path = args.cif_checkpoint
    if not os.path.exists(ckpt_path):
        # thử tìm cạnh model_checkpoint
        cand = os.path.join(args.model_checkpoint, ckpt_path)
        if os.path.exists(cand):
            ckpt_path = cand

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[CIF] checkpoint not found: {args.cif_checkpoint}")

    obj = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(obj)

    # infer conv dims/kernel
    conv_w = None
    for k, v in sd.items():
        if k.endswith("conv.weight") and isinstance(v, torch.Tensor) and v.ndim == 3:
            conv_w = v
            break

    if conv_w is not None:
        out_c, in_c, ksz = conv_w.shape
        in_dim = in_c
        conv_out = out_c
        kernel = ksz
    else:
        in_dim = encoder_out_dim
        conv_out = encoder_out_dim
        kernel = 3

    head = CIFAlphaHead(in_dim=in_dim, conv_out_dim=conv_out, kernel_size=kernel)
    missing, unexpected = head.load_state_dict(sd, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head.to(device).eval()

    if args.cif_debug:
        print(f"[CIF] loaded alpha head from: {ckpt_path}", file=sys.stderr)
        print(f"[CIF] conv(in={in_dim}, out={conv_out}, k={kernel})", file=sys.stderr)
        if missing:
            print(f"[CIF] missing keys: {missing}", file=sys.stderr)
        if unexpected:
            print(f"[CIF] unexpected keys: {unexpected}", file=sys.stderr)

    return head


def cif_integrate_and_fire(
    H_tc: torch.Tensor,
    alpha_t: torch.Tensor,
    threshold: float = 1.0,
    residual_alpha: float = 0.0,
    residual_vec: torch.Tensor | None = None,
    normalize_by_threshold: bool = True,
):
    """
    CIF integrate-and-fire "chuẩn":
      Input:
        H_tc:    (T,C) encoder outputs
        alpha_t: (T,)  weights in (0,1)
        threshold: firing threshold (default 1.0)
        residual_alpha/residual_vec: carry-over từ chunk trước (streaming)

      Output:
        C_nc: (N,C) integrated embeddings c_i (N fires in this chunk) OR None if no fire
        new_residual_alpha: float
        new_residual_vec: (C,)
        fire_positions: list[int] (frame idx nơi fire xảy ra)
    """
    assert H_tc.ndim == 2
    assert alpha_t.ndim == 1
    T, C = H_tc.shape
    device = H_tc.device
    dtype = H_tc.dtype

    if residual_vec is None:
        residual_vec = torch.zeros(C, device=device, dtype=dtype)

    acc_a = float(residual_alpha)
    acc_v = residual_vec
    emitted = []
    fires = []

    # scale factor để giữ magnitude ổn định (khi threshold != 1)
    def w_scale(w: float) -> float:
        if not normalize_by_threshold:
            return w
        return w / float(threshold)

    for t in range(T):
        a = float(alpha_t[t].item())
        h = H_tc[t]

        # alpha có thể được scale trước đó; vẫn giữ an toàn
        if a <= 0.0:
            continue

        # Một frame có thể (hiếm) tạo nhiều fire nếu a lớn và threshold nhỏ -> xử lý bằng while
        while a > 1e-8:
            need = float(threshold) - acc_a

            # chưa đủ để fire
            if a < need - 1e-8:
                acc_v = acc_v + (w_scale(a) * h)
                acc_a += a
                a = 0.0
                break

            # đủ để kết thúc 1 token (fire)
            use = need
            acc_v = acc_v + (w_scale(use) * h)
            emitted.append(acc_v)
            fires.append(t)

            # reset để tích token mới
            a -= use
            acc_a = 0.0
            acc_v = torch.zeros(C, device=device, dtype=dtype)

    if len(emitted) == 0:
        return None, acc_a, acc_v, fires

    C_nc = torch.stack(emitted, dim=0)  # (N,C)
    return C_nc, acc_a, acc_v, fires


# =========================
# ASR worker (CIF -> c_i -> CTC)
# =========================
@torch.no_grad()
def asr_worker(args, asr_model, char_dict, hypothesis_queue, cif_alpha_head: nn.Module | None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subsampling = asr_model.encoder.embed.subsampling_factor
    num_layers  = asr_model.encoder.num_blocks
    conv_lorder = asr_model.encoder.cnn_module_kernel // 2
    enc_steps   = max(1, int(round((args.stream_chunk_sec / 0.01) / subsampling)))
    chunk_size  = enc_steps

    left_ctx, right_ctx = args.left_context_size, args.right_context_size

    att_cache = torch.zeros(
        (num_layers, left_ctx, asr_model.encoder.attention_heads,
         asr_model.encoder._output_size * 2 // asr_model.encoder.attention_heads),
        device=device
    )
    cnn_cache = torch.zeros(
        (num_layers, asr_model.encoder._output_size, conv_lorder),
        device=device
    )
    offset    = torch.zeros(1, dtype=torch.int, device=device)

    # CIF streaming carry-over
    cif_residual_alpha = 0.0
    cif_residual_vec   = None  # allocated after first call

    sr            = args.mic_sr
    block_samples = int(args.stream_chunk_sec * sr)
    lookahead_blk = max(0, int(math.ceil(args.lookahead_sec / args.stream_chunk_sec)))

    q_audio  = deque(maxlen=1 + lookahead_blk)
    full_hyp = ""
    last_sent = ""

    print("ASR worker started. Listening...")
    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                            blocksize=block_samples) as stream:
            while not args.stop_event.is_set():
                audio_block, _ = stream.read(block_samples)

                q_audio.append(np.squeeze(audio_block, axis=1).astype(np.float32, copy=True))
                if len(q_audio) < 1 + lookahead_blk:
                    continue

                seg_np = np.concatenate(q_audio, dtype=np.float32)
                seg = torch.from_numpy(seg_np).unsqueeze(0).to(device) * 32768.0
                if seg.size(1) < int(0.025 * sr):
                    continue

                x = kaldi.fbank(
                    seg,
                    num_mel_bins=80,
                    frame_length=25,
                    frame_shift=10,
                    dither=0.0,
                    energy_floor=0.0,
                    sample_frequency=sr
                ).unsqueeze(0).to(device)
                x_len = torch.tensor([x.size(1)], dtype=torch.int, device=device)

                use_cuda = device.type == "cuda"
                ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16) \
                    if use_cuda else contextlib.nullcontext()

                with ctx:
                    enc_out, enc_len, _, att_cache, cnn_cache, offset = \
                        asr_model.encoder.forward_parallel_chunk(
                            xs=x,
                            xs_origin_lens=x_len,
                            chunk_size=chunk_size,
                            left_context_size=left_ctx,
                            right_context_size=right_ctx,
                            att_cache=att_cache,
                            cnn_cache=cnn_cache,
                            truncated_context_size=chunk_size,
                            offset=offset
                        )

                    T = int(enc_len.item())
                    enc_out = enc_out.reshape(1, -1, enc_out.size(-1))[:, :T]
                    if enc_out.size(1) > chunk_size:
                        enc_out = enc_out[:, :chunk_size]
                    offset = offset - T + enc_out.size(1)

                    # -----------------------------
                    # CIF proper: alpha -> I&F -> c_i
                    # -----------------------------
                    if cif_alpha_head is None:
                        # fallback: legacy behavior (CTC per frame) nếu bạn chưa có cif_best.pt
                        hyp_step = asr_model.encoder.ctc_forward(enc_out).squeeze(0).cpu()
                        chunk_text = get_output([hyp_step], char_dict)[0]
                    else:
                        # alpha_t: (T,)
                        alpha_bt = cif_alpha_head(enc_out)  # (1,T)
                        alpha_t  = alpha_bt[0]

                        # optional scaling (nếu bạn muốn giữ đúng average quantity như lúc train)
                        if args.cif_alpha_scale != 1.0:
                            alpha_t = alpha_t * float(args.cif_alpha_scale)

                        # integrate-and-fire trên frame-level H
                        H_tc = enc_out[0]  # (T,C)
                        C_nc, cif_residual_alpha, cif_residual_vec, fires = cif_integrate_and_fire(
                            H_tc=H_tc,
                            alpha_t=alpha_t,
                            threshold=args.cif_threshold,
                            residual_alpha=cif_residual_alpha,
                            residual_vec=cif_residual_vec,
                            normalize_by_threshold=True,
                        )

                        if args.cif_debug:
                            a_sum = float(alpha_t.detach().float().sum().item())
                            n_fire = 0 if C_nc is None else int(C_nc.size(0))
                            print(
                                f"\n[CIF] alpha_sum={a_sum:.2f} fires={n_fire} residual={cif_residual_alpha:.3f}",
                                file=sys.stderr
                            )

                        # nếu chunk này chưa fire được token nào -> chưa emit
                        if C_nc is None or C_nc.size(0) == 0:
                            continue

                        # chạy CTC trên chuỗi c_i (1, N, C)
                        logits = asr_model.encoder.ctc_forward(C_nc.unsqueeze(0)).squeeze(0).cpu()
                        chunk_text = get_output([logits], char_dict)[0]

                # ghép text chống lặp (vẫn giữ để an toàn)
                ovl = longest_suffix_prefix_overlap(full_hyp, chunk_text)
                if ovl < len(chunk_text):
                    full_hyp += chunk_text[ovl:]

                if full_hyp != last_sent:
                    hypothesis_queue.put(full_hyp)
                    last_sent = full_hyp

    except Exception as e:
        print("\n!!!!!!!!!! CRASH TRONG ASR WORKER !!!!!!!!!!", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print(f"\nError in ASR worker: {e}", file=sys.stderr)
    finally:
        hypothesis_queue.put(None)
        print("ASR worker finished.")


# =========================
# Model init
# =========================
@torch.no_grad()
def init_asr_model(args):
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(max(1, args.cpu_threads))

    cfg   = os.path.join(args.model_checkpoint, "config.yaml")
    ckpt  = os.path.join(args.model_checkpoint, "pytorch_model.bin")
    vocab = os.path.join(args.model_checkpoint, "vocab.txt")

    with open(cfg, "r") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    model = init_model(config, cfg)
    model.eval()
    load_checkpoint(model, ckpt)

    if torch.cuda.is_available():
        model.encoder = model.encoder.cuda()
        model.ctc     = model.ctc.cuda()

    symtab = read_symbol_table(vocab)
    char_dict = {v: k for k, v in symtab.items()}
    return model, char_dict


def init_punctuation_model(args):
    print(f"Loading punctuation model: {args.punc_model}")
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["ORT_NUM_THREADS"] = str(args.cpu_threads)

    m = PunctCapSegModelONNX.from_pretrained(args.punc_model)

    prov = ["CPUExecutionProvider"]
    if args.punc_device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif args.punc_device == "cuda":
        print("Warning: CUDAExecutionProvider not found, falling back to CPU for punctuation.",
              file=sys.stderr)

    print("Punctuation model ready on",
          "CUDA" if prov[0].startswith("CUDA") else "CPU")
    return m


# =========================
# ITN
# =========================
def init_itn_model(itn_model_dir):
    print(f"Loading ITN model from: {itn_model_dir}")
    far_dir = os.path.join(itn_model_dir, "far")
    classifier_far = os.path.join(far_dir, "classify", "tokenize_and_classify.far")
    verbalizer_far = os.path.join(far_dir, "verbalize", "verbalize.far")

    if not os.path.exists(classifier_far) or not os.path.exists(verbalizer_far):
        print(f"LỖI: Không tìm thấy file .far trong {far_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        reader_classifier = pynini.Far(classifier_far)
        reader_verbalizer = pynini.Far(verbalizer_far)
        classifier = reader_classifier.get_fst()
        verbalizer = reader_verbalizer.get_fst()
        print("ITN model ready.")
        return classifier, verbalizer
    except Exception as e:
        print(f"Lỗi khi tải ITN model: {e}", file=sys.stderr)
        sys.exit(1)


def inverse_normalize(s: str, classifier, verbalizer) -> str:
    if not s.strip():
        return s
    try:
        token = top_rewrite(s, classifier)
        return top_rewrite(token, verbalizer)
    except rewrite.Error:
        return s
    except Exception as e:
        print(f"\nError during ITN: {e}", file=sys.stderr)
        return s


def safe_itn_commit(text: str, classifier, verbalizer) -> str:
    norm = inverse_normalize(text, classifier, verbalizer)
    if has_mixed_num_span(norm):
        return text
    return norm


# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Realtime ASR + Punctuation + SAFE ITN + CIF(I&F->c_i->CTC)")

    # ASR
    ap = parser.add_argument_group("ASR")
    ap.add_argument("--model_checkpoint", type=str, required=True)
    ap.add_argument("--mic_sr", type=int, default=16000)
    ap.add_argument("--stream_chunk_sec", type=float, default=0.5)
    ap.add_argument("--lookahead_sec", type=float, default=0.5)
    ap.add_argument("--left_context_size", type=int, default=128)
    ap.add_argument("--right_context_size", type=int, default=32)
    ap.add_argument("--cpu_threads", type=int, default=1)

    # CIF
    cp = parser.add_argument_group("CIF")
    cp.add_argument("--cif_checkpoint", type=str, default="cif_best.pt",
                    help="CIF alpha head checkpoint (default: cif_best.pt)")
    cp.add_argument("--cif_threshold", type=float, default=1.0,
                    help="Integrate-and-fire threshold (default 1.0)")
    cp.add_argument("--cif_alpha_scale", type=float, default=1.0,
                    help="Optional alpha scaling at inference (default 1.0). Keep 1.0 if unsure.")
    cp.add_argument("--cif_disable", action="store_true",
                    help="Disable CIF and fallback to legacy CTC-per-frame decoding.")
    cp.add_argument("--cif_debug", action="store_true",
                    help="Print CIF diagnostics to stderr.")

    # Punctuation
    pp = parser.add_argument_group("Punctuation")
    pp.add_argument("--punc_model", type=str,
                    default="1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase")
    pp.add_argument("--punc_device", type=str, default="cuda", choices=["cuda", "cpu"])
    pp.add_argument("--use_sbd", action="store_true")
    pp.add_argument("--punc_window_words", type=int, default=240)
    pp.add_argument("--punc_commit_margin_words", type=int, default=120)
    pp.add_argument("--punc_processing_window_words", type=int, default=400)
    pp.add_argument("--punc_context_overlap_words", type=int, default=3)

    # ITN
    ip = parser.add_argument_group("ITN")
    ip.add_argument("--itn_model_dir", type=str, required=True,
                    help="Đường dẫn tới repo Vietnamese-Inverse-Text-Normalization")

    args = parser.parse_args()
    args.stop_event = threading.Event()

    # init models
    asr_model, char_dict = init_asr_model(args)
    punc_model = init_punctuation_model(args)
    itn_classifier, itn_verbalizer = init_itn_model(args.itn_model_dir)

    # init CIF
    cif_alpha_head = None
    if not args.cif_disable:
        try:
            cif_alpha_head = init_cif_alpha_head(args, encoder_out_dim=asr_model.encoder._output_size)
        except Exception as e:
            print(f"[CIF] init failed ({e}). Fallback to legacy CTC-per-frame.", file=sys.stderr)
            cif_alpha_head = None

    hyp_q = queue.Queue(maxsize=8)
    t = threading.Thread(
        target=asr_worker,
        args=(args, asr_model, char_dict, hyp_q, cif_alpha_head),
        daemon=True
    )
    t.start()

    # STATE
    raw_text        = ""
    committed_ptr   = 0
    committed_raw   = ""
    committed_norm  = ""
    last_render     = ""

    print("\nMic streaming. Ctrl+C to stop.")
    try:
        while True:
            try:
                item = hyp_q.get(timeout=0.1)
                if item is None:
                    args.stop_event.set()
                    break
                latest = item
            except queue.Empty:
                continue

            if latest == raw_text:
                continue
            raw_text = latest

            tail_raw = raw_text[committed_ptr:]
            if not tail_raw.strip():
                continue

            tokens_tail = _TOKEN_RE.findall(tail_raw)
            if not tokens_tail:
                continue

            punct_tail = punc_model.infer([tail_raw], apply_sbd=args.use_sbd)[0]

            commit_prefix, active_raw = split_commit_region(
                punct_tail,
                window_words=args.punc_window_words,
                margin_words=args.punc_commit_margin_words
            )

            if commit_prefix:
                committed_raw += commit_prefix

                commit_word_count = len(WORD_RE.findall(commit_prefix))
                committed_ptr = advance_pointer_by_words(
                    raw_text, committed_ptr, commit_word_count
                )

                commit_norm = safe_itn_commit(commit_prefix.strip(), itn_classifier, itn_verbalizer)
                if committed_norm and not committed_norm.endswith(" "):
                    committed_norm += " "
                committed_norm += commit_norm
                if not committed_norm.endswith(" "):
                    committed_norm += " "

            if active_raw.strip():
                active_norm = inverse_normalize(active_raw, itn_classifier, itn_verbalizer)
            else:
                active_norm = ""

            prefix_display = committed_norm
            if prefix_display and not prefix_display.endswith(" "):
                prefix_display += " "

            active_tokens_norm = _TOKEN_RE.findall(active_norm)
            tail_color = max(1, int(args.punc_commit_margin_words))

            if len(active_tokens_norm) > tail_color:
                head_norm = " ".join(active_tokens_norm[:-tail_color])
                tail_norm = " ".join(active_tokens_norm[-tail_color:])
                display = f"\r{prefix_display}{BLUE}{head_norm} {YELLOW}{tail_norm} "
            else:
                display = f"\r{prefix_display}{YELLOW}{' '.join(active_tokens_norm)} "

            if display != last_render:
                print(display.ljust(120), end="", flush=True)
                last_render = display

    except KeyboardInterrupt:
        print("\nStopping...")
        args.stop_event.set()
    except Exception:
        print("\n!!!!!!!!!! CRASH TRONG MAIN THREAD !!!!!!!!!!", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        args.stop_event.set()
    finally:
        t.join()

        tail_raw = raw_text[committed_ptr:]
        final_punc_tail = ""

        if tail_raw.strip():
            try:
                final_punc_tail = punc_model.infer([tail_raw], apply_sbd=args.use_sbd)[0]
            except Exception as e:
                print(f"\nError final punctuation tail: {e}", file=sys.stderr)
                final_punc_tail = tail_raw

        full_punc_text = (committed_raw + " " + final_punc_tail).strip()
        try:
            final_norm = safe_itn_commit(full_punc_text, itn_classifier, itn_verbalizer)
        except Exception as e:
            print(f"\nError final ITN: {e}", file=sys.stderr)
            final_norm = full_punc_text

        print(f"\r{final_norm.strip()} ")
        print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        itn_repo_path = ITN_REPO

        sys.argv = [
            "realtime_decode.py",
            "--model_checkpoint", CHUNKFORMER_CHECKPOINT,
            "--itn_model_dir", itn_repo_path,
            "--punc_device", "cuda",
            "--cpu_threads", "1",

            "--stream_chunk_sec", "0.5",
            "--lookahead_sec", "0.2",
            "--left_context_size", "128",
            "--right_context_size", "32",

            "--punc_processing_window_words", "400",
            "--punc_window_words", "240",
            "--punc_commit_margin_words", "120",
            "--punc_context_overlap_words", "3",

            # CIF
            "--cif_checkpoint", "cif_best.pt",
            "--cif_threshold", "1.0",
            "--cif_alpha_scale", "1.0",
            # "--cif_debug",
        ]

        if not os.path.isdir(os.path.join(itn_repo_path, "far")):
            print("=" * 50, file=sys.stderr)
            print(f"LỖI: Không tìm thấy thư mục 'far' tại: {itn_repo_path}", file=sys.stderr)
            sys.exit(1)

    main()
