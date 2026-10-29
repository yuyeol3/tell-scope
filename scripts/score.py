"""Scores texts with a DetectorScope detector and prints their evidence as JSON.

The output is what a rewriting agent needs: the verdict, how far the logit
sits above the decision threshold, and the features that push toward AI and
toward human, each with the passages where it fires marked as [[...]].
Span extraction follows spans_for() in detectorscope's
scripts/paraphrase_ablation.py.

English uses the released open-domain head. Korean uses heads/ko.json
(scripts/train_ko_head.py) over the same SAE features, on markdown-stripped
text as it was trained, through the slim model of ko_model.py (built on first
use). Several files share one model load.

  python scripts/score.py draft.txt
  python scripts/score.py v1.txt v2.txt --lang ko
"""

import argparse
import gc
import json
import os
import re
import sys

import numpy as np
import torch
from detectorscope import AuditableDetector
from detectorscope.model.detectors.auditable import Feature

import ko_model

BASE_MODEL = "liamdugan/detector-scope"
# The Korean head is tied to these SAEs; pin them so a Hub update cannot break it.
BASE_REVISION = "9baaff9e2d1cdb79cafa90cdb27811b847e69dae"
HOOK_LAYERS = (8, 14, 18, 23)  # blocks whose inputs the SAEs read, in feature-id order
D_SAE = 16384  # features per SAE; feature id = SAE index * D_SAE + index within it
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


# One emoji with its variation selectors, skin tones and ZWJ-joined parts.
EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF](?:[\uFE0F\U0001F3FB-\U0001F3FF]|\u200D[\U0001F000-\U0001FAFF\u2600-\u27BF])*")

# Tells caught by rule rather than by the detector: punctuation and emoji the
# Korean head never saw in training (emoji even read as human there), and stock
# phrases. English words follow DetectorScope's para_antitell prompt.
RULES = [
    ("em-dash", re.compile(r"\s?[—–]\s?"), "쉼표, 괄호, 콜론으로 바꾸거나 문장을 끊는다 / use a comma, colon or a new sentence"),
    ("middle-dot", re.compile(r"·"), "'와/과'나 쉼표로 풀어 쓴다"),
    ("emoji", EMOJI, "장식 이모지는 뺀다 / drop decorative emoji"),
    ("stock-phrase", re.compile(
        r"오늘날|빠르게 변화하는|그 어느 때보다|단순히 .{1,20}?(을|를) 넘어|단순한 .{1,15}?(이|가) 아닌|그렇다면|"
        r"(에 대해|을|를) (자세히 |함께 )?알아보(겠습니다|도록 하겠습니다|자)|결론적으로|요약하자면|마무리하며|"
        r"것으로 기대(됩니다|된다)|극대화|유기적으로|진정한|여러분의 .{1,10}?응원"),
     "상투구다. 구체적인 사실이나 자기 말로 바꾸거나 뺀다"),
    ("stock-phrase", re.compile(
        r"\b(delve|crucial|vital|pivotal|landscape|comprehensive|robust|leverag\w*|foster\w*|navigat\w*|testament|"
        r"tapestry|underscor\w*|showcas\w*|seamless\w*|multifaceted|In today's|fast-paced)\b|"
        r"(^|[.!?]\s+)(Furthermore|Additionally|Moreover|Ultimately|Overall|In conclusion|Notably|Importantly),",
        re.I),
     "stock AI vocabulary or opener; say it plainly or cut it"),
]


def rule_tells(text: str) -> list[dict]:
    """Every rule match, in text order, with a line of context."""
    out = []
    for name, pattern, fix in RULES:
        for m in pattern.finditer(text):
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.end())
            line = text[line_start : None if line_end < 0 else line_end].strip()
            out.append({"rule": name, "match": m.group(0).strip(), "line": line[:120], "fix": fix, "pos": m.start()})
    return [{k: v for k, v in t.items() if k != "pos"} for t in sorted(out, key=lambda t: t["pos"])]


def plain(text: str) -> str:
    """Markdown and emoji stripped, newlines normalized, as the Korean head was trained.

    Without this the head separates formats (CRLF, '#' headers) rather than
    writing styles. KatFish has no emoji, and one rare-character feature read
    them as human.
    """
    text = EMOJI.sub("", text)
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


def ko_detect(detector, head: dict, text: str) -> tuple[float, list[Feature], list[str]]:
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


def score(models: dict, text: str, lang: str, ko_head: dict | None, args) -> dict:
    """`models` maps a language to its extractor: the full detector, or for "ko" the slim KoModel."""
    meta = {}
    if lang == "ko":
        logit, feats, tokens = ko_detect(models["ko"], ko_head, text)
        threshold = ko_head["threshold"]
        meta = {f["id"]: f for f in ko_head["features"]}
    else:
        detector = models["en"]
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
        "tells": rule_tells(text),
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
    models = {}
    if "ko" in langs:
        if not ko_model.matches(ko_head):  # one-time: cut the slim model out of the full detector
            ko_model.build(load(), ko_head)
            gc.collect()
            torch.cuda.empty_cache()
        models["ko"] = ko_model.KoModel()
    if "en" in langs:
        models["en"] = load()
    results = [dict(path=path, **score(models, t, lang, ko_head, args)) for path, t, lang in zip(args.paths, texts, langs)]
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
