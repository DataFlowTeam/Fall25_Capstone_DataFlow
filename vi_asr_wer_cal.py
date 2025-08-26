
#!/usr/bin/env python3
# vi_asr_eval.py
# Vietnamese text normalization + WER/CER for ASR evaluation.
# Usage examples:
#   python vi_asr_eval.py --ref refs.txt --hyp hyps.txt
#   python vi_asr_eval.py --pairs pairs.tsv --columns ref,hyp --agg aggressive
#   python vi_asr_eval.py --ref refs.txt --hyp hyps.txt --lower --rm-punct --map-digits
#
# File formats:
#   refs.txt / hyps.txt: one utterance per line in the same order.
#   pairs.tsv: tab-separated with named columns; select with --columns ref,hyp
#
# Exit code 0 and prints a JSON with metrics.
from __future__ import annotations
import sys, re, unicodedata, argparse, json, pathlib

PUNCT_PATTERN = re.compile(r'[\u0000-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E\u2000-\u206F\u2E00-\u2E7F“”‘’…]+', flags=re.UNICODE)

def normalize_vi(text: str,
                 lower: bool = True,
                 nfc: bool = True,
                 rm_punct: bool = False,
                 map_digits: bool = False,
                 collapse_space: bool = True,
                 strip_extra_quotes: bool = True,
                 keep_hyphen_inside_words: bool = True) -> str:
    """Basic, deterministic VN normalization suitable for ASR eval."""
    if text is None:
        return ""
    t = text.strip()
    if nfc:
        t = unicodedata.normalize("NFC", t)
    # Standardize fancy quotes/dashes to ASCII
    t = t.replace("“","\"").replace("”","\"").replace("‘","'").replace("’","'").replace("–","-").replace("—","-").replace("…","...")
    if strip_extra_quotes:
        t = re.sub(r"[\"'`]+", "", t)
    if keep_hyphen_inside_words:
        # remove hyphens that are not between letters or digits
        t = re.sub(r"(?<![0-9A-Za-zÀ-ỹà-ỹ])-|-(?![0-9A-Za-zÀ-ỹà-ỹ])", " ", t)
    if map_digits:
        # Replace sequences of digits with a placeholder token.
        # This avoids mismatches due to formatting ("12" vs "mười hai").
        t = re.sub(r"\d+", "<num>", t)
    if rm_punct:
        t = PUNCT_PATTERN.sub(" ", t)
    if lower:
        t = t.lower()
    if collapse_space:
        t = re.sub(r"\s+", " ", t).strip()
    return t

def _levenshtein_alignment(ref_tokens, hyp_tokens):
    # Returns distance and alignment ops for analysis.
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        ri = ref_tokens[i-1]
        for j in range(1, m+1):
            hj = hyp_tokens[j-1]
            cost = 0 if ri == hj else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    # backtrack for ops
    i, j = n, m
    ops = []
    while i>0 or j>0:
        if i>0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(("D", ref_tokens[i-1], ""))
            i -= 1
        elif j>0 and dp[i][j] == dp[i][j-1] + 1:
            ops.append(("I", "", hyp_tokens[j-1]))
            j -= 1
        else:
            op = "C" if ref_tokens[i-1] == hyp_tokens[j-1] else "S"
            ops.append((op, ref_tokens[i-1], hyp_tokens[j-1]))
            i -= 1; j -= 1
    ops.reverse()
    return dp[n][m], ops

def wer(ref: str, hyp: str) -> dict:
    ref_toks = ref.split()
    hyp_toks = hyp.split()
    dist, ops = _levenshtein_alignment(ref_toks, hyp_toks)
    N = max(1, len(ref_toks))
    return {
        "WER": dist / N,
        "N_words": len(ref_toks),
        "distance": dist,
        "ops": ops
    }

def cer(ref: str, hyp: str) -> dict:
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))
    dist, ops = _levenshtein_alignment(ref_chars, hyp_chars)
    N = max(1, len(ref_chars))
    return {
        "CER": dist / N,
        "N_chars": len(ref_chars),
        "distance": dist,
        "ops": ops
    }

def evaluate_pairs(ref_lines, hyp_lines, norm_kwargs):
    assert len(ref_lines) == len(hyp_lines), "ref and hyp must have same length"
    tot_w_dist = tot_w_N = 0
    tot_c_dist = tot_c_N = 0
    samples = []
    for idx, (r, h) in enumerate(zip(ref_lines, hyp_lines)):
        r_n = normalize_vi(r, **norm_kwargs)
        h_n = normalize_vi(h, **norm_kwargs)
        w = wer(r_n, h_n)
        c = cer(r_n, h_n)
        tot_w_dist += w["distance"]; tot_w_N += w["N_words"]
        tot_c_dist += c["distance"]; tot_c_N += c["N_chars"]
        samples.append({
            "id": idx,
            "ref": r.strip(),
            "hyp": h.strip(),
            "ref_norm": r_n,
            "hyp_norm": h_n,
            "WER": w["WER"],
            "CER": c["CER"],
            "ops_word": w["ops"][:50],  # truncate
            "ops_char": c["ops"][:50]
        })
    out = {
        "micro_WER": (tot_w_dist / max(1, tot_w_N)),
        "micro_CER": (tot_c_dist / max(1, tot_c_N)),
        "total_words": tot_w_N,
        "total_chars": tot_c_N,
        "n_utt": len(ref_lines),
        "norm": norm_kwargs,
        "samples_preview": samples[:5]
    }
    return out

def read_list_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def read_pairs_tsv(path, ref_col, hyp_col):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append(parts)
    # header?
    header = rows[0]
    if ref_col in header and hyp_col in header:
        colmap = {name:i for i,name in enumerate(header)}
        body = rows[1:]
        refs = [r[colmap[ref_col]] for r in body]
        hyps = [r[colmap[hyp_col]] for r in body]
    else:
        # treat as positional indices "0,1"
        ri, hi = map(int, (ref_col, hyp_col))
        refs = [r[ri] for r in rows]
        hyps = [r[hi] for r in rows]
    return refs, hyps

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pairs", type=str, help="TSV file with reference and hypothesis columns")
    g.add_argument("--ref", type=str, help="Path to refs.txt (one per line)")
    ap.add_argument("--hyp", type=str, help="Path to hyps.txt (one per line)")
    ap.add_argument("--columns", type=str, default="ref,hyp", help="When using --pairs, specify 'ref_col,hyp_col' by name or index")
    ap.add_argument("--agg", choices=["basic","aggressive"], default="basic",
                    help="Preset for normalization")
    ap.add_argument("--lower", action="store_true", help="Lowercase")
    ap.add_argument("--rm-punct", action="store_true", help="Remove punctuation")
    ap.add_argument("--map-digits", action="store_true", help="Map digit sequences to <num>")
    ap.add_argument("--no-nfc", action="store_true", help="Disable NFC normalization")
    ap.add_argument("--json", action="store_true", help="Print JSON only")
    args = ap.parse_args()

    # Presets
    if args.agg == "basic":
        norm_kwargs = dict(lower=True, nfc=True, rm_punct=False, map_digits=False,
                           collapse_space=True, strip_extra_quotes=True, keep_hyphen_inside_words=True)
    else:
        norm_kwargs = dict(lower=True, nfc=True, rm_punct=True, map_digits=True,
                           collapse_space=True, strip_extra_quotes=True, keep_hyphen_inside_words=True)
    # Overrides
    if args.lower: norm_kwargs["lower"] = True
    if args.rm_punct: norm_kwargs["rm_punct"] = True
    if args.map_digits: norm_kwargs["map_digits"] = True
    if args.no_nfc: norm_kwargs["nfc"] = False

    if args.pairs:
        ref_col, hyp_col = args.columns.split(",")
        refs, hyps = read_pairs_tsv(args.pairs, ref_col.strip(), hyp_col.strip())
    else:
        if not args.hyp:
            print("Error: --hyp is required when using --ref", file=sys.stderr)
            sys.exit(2)
        refs = read_list_file(args.ref)
        hyps = read_list_file(args.hyp)
    out = evaluate_pairs(refs, hyps, norm_kwargs)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())