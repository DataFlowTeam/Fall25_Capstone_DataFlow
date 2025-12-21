
# ========================================================
"""
CIF inference ONLY (streaming)
- Removed: CIF training loop, dataset loading, optimizer, checkpoint saving, legacy decode.
- Kept: encoder streaming -> CIF firing -> span-wise frame-CTC decode (with boundary memory).
"""
from typing import Any, List, Tuple
from torch import nn
from model.cif_head import CifCtcHead
import os
import torch
import yaml
from pydub import AudioSegment
from model.utils.init_model import init_model
from model.utils.checkpoint import load_checkpoint
from model.utils.file_utils import read_symbol_table
from model.cif import CifMiddleware
# ========================================================


# ---------------- CIF Config ----------------
class CIFCfg:
    # bạn có thể đưa các giá trị này từ config.yaml / hoặc sẽ được override từ ckpt nếu có
    cif_threshold = 1.0
    cif_embedding_dim = 512
    encoder_embed_dim = 512
    produce_weight_type = "conv"    # "conv" | "dense" | "linear"
    conv_cif_width = 5
    conv_cif_dropout = 0.1
    apply_scaling = True
    apply_tail_handling = True
    tail_handling_firing_threshold = 0.4

# ---------------- Runner: encoder -> CIF -> CTC ----------------
class EndlessRunner(nn.Module):
    def __init__(self, model, vocab_size, blank_id, cif_cfg: CIFCfg):
        super().__init__()
        self.model = model                       # có encoder (+ có thể có CTC_enc)
        self.cif = CifMiddleware(cif_cfg)
        self.cif_ctc = CifCtcHead(cif_cfg.cif_embedding_dim, vocab_size, blank_id)

# ---------------- Utilities ----------------
def build_char_maps(char_dict):
    """
    char_dict: id->char
    """
    id2ch = char_dict
    ch2id = {ch: i for i, ch in id2ch.items()}

    blank_id = None
    for k, v in id2ch.items():
        if v in ("<blk>", "<blank>", "<ctc_blank>"):
            blank_id = k
            break
    if blank_id is None:
        blank_id = 0  # fallback

    unk_id = None
    for k, v in id2ch.items():
        if v in ("<unk>", "<UNK>"):
            unk_id = k
            break

    return id2ch, ch2id, blank_id, unk_id


def ids_to_text(ids, id2ch):
    return "".join(id2ch[i] for i in ids if i in id2ch)


def load_audio(audio_path: str) -> torch.Tensor:
    """
    Return waveform tensor: (1, T) float32 @16kHz mono
    """
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)  # 16bit
    audio = audio.set_channels(1)      # mono
    wav = torch.as_tensor(audio.get_array_of_samples(), dtype=torch.float32).unsqueeze(0)
    return wav


@torch.no_grad()
def init_asr_model(model_checkpoint: str, device: torch.device):
    """
    Load your repo model (encoder + optional ctc head inside encoder).
    """
    config_path = os.path.join(model_checkpoint, "config.yaml")
    checkpoint_path = os.path.join(model_checkpoint, "pytorch_model.bin")
    symbol_table_path = os.path.join(model_checkpoint, "vocab.txt")

    with open(config_path, "r") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    model = init_model(config, config_path)
    model.eval()
    load_checkpoint(model, checkpoint_path)

    model.encoder = model.encoder.to(device)
    if hasattr(model, "ctc"):
        model.ctc = model.ctc.to(device)

    symbol_table = read_symbol_table(symbol_table_path)
    # symbol_table: char -> id
    # char_dict cần id -> char
    char_dict = {v: k for k, v in symbol_table.items()}
    return model, char_dict


def _apply_cif_cfg_from_ckpt(cif_cfg: CIFCfg, ckpt: dict):
    """
    If your CIF ckpt was saved by save_cif_checkpoint(), it contains a cif_cfg dict.
    We'll apply it (except dims which will be overridden by encoder output size).
    """
    cfg = ckpt.get("cif_cfg", None)
    if not isinstance(cfg, dict):
        return
    for k, v in cfg.items():
        if hasattr(cif_cfg, k):
            setattr(cif_cfg, k, v)


def load_cif_from_ckpt(cif: CifMiddleware, ckpt_path: str, device: torch.device) -> dict:
    """
    Load CIF state dict from checkpoint.
    Returns loaded ckpt dict for optional metadata (cfg).
    """
    if not (ckpt_path and os.path.isfile(ckpt_path)):
        raise FileNotFoundError(f"Invalid --cif_ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "cif" in ckpt:
        cif.load_state_dict(ckpt["cif"], strict=True)
    else:
        # fallback: treat file as plain state_dict
        cif.load_state_dict(ckpt, strict=True)
    return ckpt



# ---------------- CIF Boundary (global singleton) ----------------
# split_encoder_outs_by_cif_fires() chỉ nhận (encoder_outs, cif_carry),
# nên CIF module sẽ được giữ ở dạng singleton trong file này.
_CIF_BOUNDARY = {
    "cif": None,          # type: CifMiddleware | None
    "device": None,       # type: torch.device | None
    "cif_cfg": None,      # type: CIFCfg | None
    "ckpt_path": None,    # type: str | None
}


def set_cif_boundary(cif: CifMiddleware):
    """Gắn CIF module (đã load weights) vào singleton để dùng trong split."""
    _CIF_BOUNDARY["cif"] = cif
    _CIF_BOUNDARY["device"] = next(cif.parameters()).device if any(True for _ in cif.parameters()) else None


@torch.no_grad()
def init_cif_boundary_from_ckpt(
    cif_ckpt_path: str,
    encoder_out_dim: int,
    device: str | torch.device = "cuda",
) -> CifMiddleware:
    """
    Init CIF boundary module từ ckpt và lưu vào singleton.
    Bạn chỉ cần gọi 1 lần trước khi bắt đầu streaming.

    - encoder_out_dim: = encoder_outs.shape[-1] của encoder.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device

    cif_cfg = CIFCfg()
    # dim phải khớp encoder_outs
    cif_cfg.encoder_embed_dim = int(encoder_out_dim)
    cif_cfg.cif_embedding_dim = int(encoder_out_dim)

    # load ckpt để override cfg (threshold, tail handling, produce_weight_type, ...)
    ckpt = torch.load(cif_ckpt_path, map_location=dev)
    if isinstance(ckpt, dict):
        _apply_cif_cfg_from_ckpt(cif_cfg, ckpt)

    cif = CifMiddleware(cif_cfg).to(dev)
    load_cif_from_ckpt(cif, cif_ckpt_path, dev)
    cif.eval()

    _CIF_BOUNDARY["cif"] = cif
    _CIF_BOUNDARY["device"] = dev
    _CIF_BOUNDARY["cif_cfg"] = cif_cfg
    _CIF_BOUNDARY["ckpt_path"] = cif_ckpt_path
    return cif


def _unpack_cif_carry(cif_carry: Any) -> tuple[Any, bool]:
    """
    Vì split() chỉ có 2 input, ta encode flush_tail vào cif_carry dạng dict:
        cif_carry = {"carry": <carry>, "flush_tail": True/False}

    Nếu cif_carry không phải dict => hiểu là carry thuần, flush_tail=False.
    """
    if isinstance(cif_carry, dict) and ("carry" in cif_carry or "flush_tail" in cif_carry):
        return cif_carry.get("carry", None), bool(cif_carry.get("flush_tail", False))
    return cif_carry, False


def _pack_cif_carry(carry: Any, flush_tail: bool = False):
    """Đóng gói carry để dễ pass giữa các chunk (và có thể đánh dấu flush_tail)."""
    return {"carry": carry, "flush_tail": bool(flush_tail)}


def mark_flush_tail(cif_carry: Any):
    """Helper: đánh dấu chunk hiện tại là chunk cuối (flush tail)."""
    carry, _ = _unpack_cif_carry(cif_carry)
    return _pack_cif_carry(carry, flush_tail=True)


def ctc_greedy_collapse_with_memory(frame_ids: torch.Tensor, blank_id: int, last_nonblank_id: int | None):
    """
    frame_ids: (L,) frame-level argmax ids
    Do classic CTC greedy collapse per span, then stitch boundary duplicates with last_nonblank_id.
    """
    out = []
    prev = None
    for t in frame_ids.tolist():
        if t == blank_id:
            prev = None
            continue
        if t != prev:
            out.append(t)
            prev = t

    if out and last_nonblank_id is not None and out[0] == last_nonblank_id:
        out = out[1:]

    new_last = out[-1] if out else last_nonblank_id
    return out, new_last


@torch.no_grad()
def split_encoder_outs_by_cif_fires(
    encoder_outs: torch.Tensor,   # (1, Te, C)
    cif_carry: Any = None,        # carry state của CIF (streaming) hoặc dict {"carry":..., "flush_tail":...}
) -> Tuple[List[torch.Tensor], Any, List[Tuple[int, int]]]:
    """
    Input: encoder_outs của 1 chunk
    Output:
      - seg_feats_list: list các segment tensor (mỗi cái shape: (1, L, C))
      - new_cif_carry: carry mới để truyền sang chunk sau (giữ format giống cif_carry input)
      - spans: list (s, e) inclusive để bạn debug/align timestamps nếu muốn

    NOTE:
      - Vì signature chỉ có (encoder_outs, cif_carry), nếu muốn flush tail ở chunk cuối
        hãy truyền cif_carry dạng dict: {"carry": carry, "flush_tail": True}
        hoặc dùng helper mark_flush_tail(cif_carry).
    """
    assert encoder_outs.dim() == 3, f"encoder_outs must be (B, T, C), got {encoder_outs.shape}"
    B, Te, _ = encoder_outs.shape
    assert B == 1, "Hàm này đang viết cho streaming batch=1."

    cif = _CIF_BOUNDARY.get("cif", None)
    if cif is None:
        raise RuntimeError(
            "CIF boundary chưa init. Hãy gọi init_cif_boundary_from_ckpt(...) "
            "hoặc set_cif_boundary(...) trước khi gọi split_encoder_outs_by_cif_fires()."
        )

    device = encoder_outs.device
    carry_in, flush_tail = _unpack_cif_carry(cif_carry)

    # --- CIF ---
    encoder_padding_mask = torch.zeros(B, Te, dtype=torch.bool, device=device)
    cif_in = {"encoder_raw_out": encoder_outs, "encoder_padding_mask": encoder_padding_mask}

    cif_pack = cif(
        cif_in,
        target_lengths=None,
        carry=carry_in,
        flush_tail=flush_tail,
    )
    carry_out = cif_pack.get("carry", None)
    fires = cif_pack["fire_mask_per_frame"][0].bool()  # (Te,)

    # --- Fire -> spans ---
    spans: List[Tuple[int, int]] = []
    seg_start = 0
    for i, f in enumerate(fires.tolist()):
        if f:
            spans.append((seg_start, i))  # inclusive
            seg_start = i + 1

    # --- spans -> seg_feats ---
    seg_feats_list: List[torch.Tensor] = []
    for (s, e) in spans:
        if e < s:
            continue
        seg_feats = encoder_outs[:, s:e + 1, :]  # (1, L, C)
        seg_feats_list.append(seg_feats)

    # giữ format giống input (dict hay carry thuần)
    if isinstance(cif_carry, dict):
        new_cif_carry = _pack_cif_carry(carry_out, flush_tail=False)
    else:
        new_cif_carry = carry_out

    return seg_feats_list, new_cif_carry, spans
