"""Scores texts with a DetectorScope detector and prints their evidence as JSON.

The output is what a rewriting agent needs: the verdict, how far the logit
sits above the decision threshold, and the features that push toward AI and
toward human, each with the passages where it fires marked as [[...]].
Span extraction follows spans_for() in detectorscope's
scripts/paraphrase_ablation.py.

English uses the released open-domain head. Korean uses heads/ko.json
(scripts/train_ko_head.py) over the same SAE features, on markdown-stripped
text as it was trained. Several files share one model load (~1 min).

  python scripts/score.py draft.txt
  python scripts/score.py v1.txt v2.txt --lang ko
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from detectorscope import AuditableDetector
from detectorscope.model.detectors.auditable import Feature

BASE_MODEL = "liamdugan/detector-scope"
# The Korean head is tied to these SAEs; pin them so a Hub update cannot break it.
BASE_REVISION = "9baaff9e2d1cdb79cafa90cdb27811b847e69dae"
KO_HEAD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "heads", "ko.json")
MAX_TOKENS = 1024  # HFExtractor truncates every text here


def load(model: str = BASE_MODEL, revision: str | None = BASE_REVISION) -> AuditableDetector:
    """Loads a detector in bf16 without the parts its SAEs never read.

    The SAEs read the input of the last hooked block, so the blocks after it
    and the vocabulary projection only cost time and memory. Dropping them
    keeps a 6 GB laptop GPU from spilling into system memory (4x faster on a
    ~900-token text, same scores).
    """
    detector = AuditableDetector.from_pretrained(model, revision=revision, torch_dtype="bfloat16")
    lm = detector.extractor.model
    last = max(int(h.rsplit(".", 1)[1]) for h in detector.hooks)
    lm.model.layers = lm.model.layers[: last + 1]
    lm.lm_head = torch.nn.Identity()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return detector


def plain(text: str) -> str:
    """Markdown stripped and newlines normalized, as the Korean head was trained.

    Without this the head separates formats (CRLF, '#' headers) rather than
    writing styles.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    out = []
    for line in text.split("\n"):
        line = re.sub(r"^\s*(#{1,6}\s+|[-*+]\s+(\[[ xX]\]\s*)?|\d+[.)]\s+|>\s*)", "", line)
        line = re.sub(r"\*\*|__|`|~~", "", line)
        line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)
        if re.fullmatch(r"\s*[-=|:]{3,}\s*", line):
            continue
        out.append(line.strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def is_korean(text: str) -> bool:
    letters = re.findall(r"[가-힣A-Za-z]", text)
    return bool(letters) and sum("가" <= c <= "힣" for c in letters) / len(letters) > 0.3


def runs(highlights: list[tuple[int, float]]) -> list[tuple[int, int, float]]:
    """Contiguous firing runs as (start, end, max activation), strongest first."""
    out = []
    for t, a in sorted(highlights):
        if out and t == out[-1][1] + 1:
            s, _, m = out[-1]
            out[-1] = (s, t, max(m, a))
        else:
            out.append((t, t, a))
    return sorted(out, key=lambda r: -r[2])


def passage(tokens: list[str], start: int, end: int, context: int) -> str:
    """One run as "...before[[core]]after...", newlines flattened."""
    before = "".join(tokens[max(0, start - context) : start])
    core = "".join(tokens[start : end + 1])
    after = "".join(tokens[end + 1 : end + 1 + context])
    return "..." + (before + "[[" + core + "]]" + after).replace("\n", " / ").strip() + "..."


def ko_detect(detector: AuditableDetector, head: dict, text: str) -> tuple[float, list[Feature], list[str]]:
    """(logit, features by |importance|, tokens) of a Korean text under heads/ko.json.

    The head is linear in log1p of the pooled activations, so a feature's
    zero-ablation importance is exactly weight * log1p(activation).
    """
    ex = detector.extract(plain(text))
    ids = torch.tensor([f["id"] for f in head["features"]])
    weights = np.array([f["weight"] for f in head["features"]])
    contrib = weights * np.log1p(ex.x[0, ids].numpy())
    cols = ex.token_feats.index_select(1, ids).coalesce()
    highlights: dict[int, list[tuple[int, float]]] = {}
    for (t, k), a in zip(cols.indices().T.tolist(), cols.values().tolist()):
        highlights.setdefault(k, []).append((t, a))
    feats = [
        Feature(f["id"], float(c), f["description"], highlights.get(k))
        for k, (f, c) in enumerate(zip(head["features"], contrib))
        if c != 0
    ]
    feats.sort(key=lambda f: -abs(f.importance))
    return head["bias"] + float(contrib.sum()), feats, ex.tokens


def feature_entry(f: Feature, tokens: list[str], n_passages: int, context: int, meta: dict | None = None) -> dict:
    """One feature for the JSON output; `meta` adds the Korean head's hint and confidence."""
    rs = runs(f.highlights or [])
    entry = {
        "id": f.id,
        "importance": round(f.importance, 3),
        "description": f.description,
        "n_passages": len(rs),
        "passages": [passage(tokens, s, e, context) for s, e, _ in rs[:n_passages]],
    }
    if meta:
        entry["confidence"] = meta.get("confidence")
        if meta.get("hint"):
            entry["hint"] = meta["hint"]
    return entry


def score(detector, text: str, lang: str, ko_head: dict | None, args) -> dict:
    meta = {}
    if lang == "ko":
        logit, feats, tokens = ko_detect(detector, ko_head, text)
        threshold = ko_head["threshold"]
        meta = {f["id"]: f for f in ko_head["features"]}
    else:
        det = detector(text, head=args.head, tau=args.tau, return_highlights=True)
        logit, feats, tokens = det.logit, det.features, det.tokens
        threshold = float(detector.load_head(args.head, args.tau).threshold)
    ai = [f for f in feats if f.importance > 0][: args.top]
    human = [f for f in feats if f.importance < 0][: args.top]
    return {
        "lang": lang,
        "label": "ai" if logit > threshold else "human",
        "p_ai": round(float(torch.sigmoid(torch.tensor(logit))), 4),
        "logit": round(logit, 3),
        "threshold": round(threshold, 3),
        # How much the logit must fall for the verdict to flip to human.
        "gap": round(logit - threshold, 3),
        "n_tokens": len(tokens),
        "truncated": len(tokens) >= MAX_TOKENS,
        "ai_features": [feature_entry(f, tokens, args.passages, args.context, meta.get(f.id)) for f in ai],
        "human_features": [feature_entry(f, tokens, args.passages, args.context, meta.get(f.id)) for f in human],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="UTF-8 text files, or - for stdin")
    p.add_argument("--lang", default="auto", choices=["auto", "en", "ko"])
    p.add_argument("--head", default="logreg", choices=["logreg", "xgboost"], help="English head type")
    p.add_argument("--tau", type=float, default=0.85, help="English head's interpretability threshold")
    p.add_argument("--top", type=int, default=8, help="features listed per side")
    p.add_argument("--passages", type=int, default=3, help="passages shown per feature")
    p.add_argument("--context", type=int, default=6, help="tokens of context around a passage")
    args = p.parse_args()

    texts = [sys.stdin.read() if path == "-" else open(path, encoding="utf-8").read() for path in args.paths]
    langs = [("ko" if is_korean(t) else "en") if args.lang == "auto" else args.lang for t in texts]
    ko_head = json.load(open(KO_HEAD, encoding="utf-8")) if "ko" in langs else None
    detector = load()
    results = [dict(path=path, **score(detector, t, lang, ko_head, args)) for path, t, lang in zip(args.paths, texts, langs)]
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
