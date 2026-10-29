"""Collects, for each feature of heads/ko.json, the Korean passages where it fires hardest.

The released descriptions were written from English text and do not hold in
Korean, so the Korean head's features are re-described from these snippets
(by an LLM reading them; see README). Uses the features cached by
train_ko_head.py, then re-extracts only the chosen documents for token
positions.

  python scripts/ko_snippets.py --katfish path/to/katfish_dataset --cache work/ \
      [--extra more.jsonl ...] --out work/ko_snippets.json

--extra files are JSON lines of {"text": ..., "label": 0|1}; their features are
extracted here.
"""

import argparse
import json
import os

import numpy as np
import torch
from scipy import sparse

from score import KO_HEAD, load, plain, runs
from train_ko_head import GENRES

CONTEXT = 15  # tokens on each side of a snippet


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--katfish", required=True)
    p.add_argument("--cache", default="work")
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--per-feature", type=int, default=8, help="documents per feature")
    p.add_argument("--out", default=os.path.join("work", "ko_snippets.json"))
    args = p.parse_args()

    head = json.load(open(KO_HEAD, encoding="utf-8"))
    det = load()

    docs, X = [], []
    for g in GENRES:
        rows = [json.loads(line) for line in open(os.path.join(args.katfish, f"{g}.jsonl"), encoding="utf-8") if line.strip()]
        docs += [{"text": r["text"], "label": r["label"], "src": f"katfish-{g}"} for r in rows]
        X.append(sparse.load_npz(os.path.join(args.cache, f"katfish_{g}.npz")))
    for path in args.extra:
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        docs += [{"text": r["text"], "label": r["label"], "src": os.path.basename(path)} for r in rows]
        X.append(sparse.vstack([sparse.csr_matrix(det.extract(plain(r["text"]), highlights=False).x.numpy()) for r in rows]))
    X = sparse.vstack(X).tocsc()
    y = np.array([d["label"] for d in docs])

    ids = [f["id"] for f in head["features"]]
    wanted: dict[int, list[int]] = {}  # document -> features to read from it
    for fid in ids:
        col = X[:, fid].toarray().ravel()
        for i in np.argsort(-col)[: args.per_feature]:
            if col[i] > 0:
                wanted.setdefault(int(i), []).append(fid)

    snippets: dict[int, list[dict]] = {fid: [] for fid in ids}
    for i, fids in wanted.items():
        ex = det.extract(plain(docs[i]["text"]))
        cols = ex.token_feats.index_select(1, torch.tensor(fids)).coalesce()
        highlights: dict[int, list[tuple[int, float]]] = {}
        for (t, k), a in zip(cols.indices().T.tolist(), cols.values().tolist()):
            highlights.setdefault(fids[k], []).append((t, a))
        for fid, h in highlights.items():
            s, e, a = runs(h)[0]
            tok = ex.tokens
            text = "".join(tok[max(0, s - CONTEXT) : s]) + "[[" + "".join(tok[s : e + 1]) + "]]" + "".join(tok[e + 1 : e + 1 + CONTEXT])
            snippets[fid].append({"act": round(a, 2), "doc_label": "ai" if docs[i]["label"] else "human",
                                  "src": docs[i]["src"], "text": text.replace("\n", " / ")})

    out = []
    for f in head["features"]:
        fires = X[:, f["id"]].toarray().ravel() > 0
        out.append({"id": f["id"], "weight": round(f["weight"], 3), "direction": "ai" if f["weight"] > 0 else "human",
                    "fires_in_ai_docs": round(float(fires[y == 1].mean()), 2),
                    "fires_in_human_docs": round(float(fires[y == 0].mean()), 2),
                    "snippets": sorted(snippets[f["id"]], key=lambda s: -s["act"])})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(out)} features, {len(wanted)} documents -> {args.out}")


if __name__ == "__main__":
    main()
