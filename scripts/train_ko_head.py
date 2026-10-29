"""Trains the Korean head over the open-domain detector's SAE features.

The released DetectorScope heads are trained on English and score Korean at
chance, but the SAE features underneath still separate Korean human and AI
text. This fits a sparse L1 logistic regression on log1p of the max-pooled
features of KatFish (https://github.com/Shinwoo-Park/katfishnet, essays and
abstracts; poetry hurt transfer to prose) and writes heads/ko.json. Feature
descriptions are added afterwards from scripts/ko_snippets.py (see README).

  python scripts/train_ko_head.py --katfish path/to/katfish_dataset --cache work/

The decision threshold maximizes Youden's J on 5-fold out-of-fold scores
unless --threshold is given. KatFish's 2024-era AI texts score far higher than
current models' (median logit +3.6 vs ~0), so its own threshold (+0.80) let
most Claude-written blog and PR texts pass as human. The released head uses
-0.5: Youden's J picked -0.52 on Namuwiki/blog pairs and -0.59 on PR pairs,
and each transferred to the other genre (TPR 0.67 FPR 0.08 on PRs, TPR 0.95
FPR 0.17 on blog). Those pairs are not redistributable, so they are not in
this repository.

The released head (87 features) was fit before liblinear was seeded, so a
rerun gives a similar but not identical feature set, whose descriptions would
have to be rewritten.
"""

import argparse
import json
import os

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

from score import BASE_MODEL, BASE_REVISION, load, plain

GENRES = ["essay", "abstract"]


def katfish_features(katfish: str, cache: str) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Pooled SAE features and labels of KatFish, extracting what the cache lacks."""
    det = None
    Xs, ys = [], []
    for g in GENRES:
        npz, meta = os.path.join(cache, f"katfish_{g}.npz"), os.path.join(cache, f"katfish_{g}.json")
        rows = [json.loads(line) for line in open(os.path.join(katfish, f"{g}.jsonl"), encoding="utf-8") if line.strip()]
        if not os.path.exists(npz):
            det = det or load(BASE_MODEL, BASE_REVISION)
            X = sparse.vstack([sparse.csr_matrix(det.extract(plain(r["text"]), highlights=False).x.numpy()) for r in rows])
            sparse.save_npz(npz, X.tocsr())
            json.dump([{"written_by": r["written_by"], "label": r["label"]} for r in rows], open(meta, "w"))
        Xs.append(sparse.load_npz(npz))
        ys += [m["label"] for m in json.load(open(meta))]
    return sparse.vstack(Xs).tocsr().log1p(), np.array(ys)


def fit(X, y, C: float) -> LogisticRegression:
    return LogisticRegression(l1_ratio=1.0, C=C, solver="liblinear", max_iter=3000, random_state=0).fit(X, y)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--katfish", required=True, help="directory with essay.jsonl and abstract.jsonl")
    p.add_argument("--cache", default="work", help="directory for extracted features")
    p.add_argument("--C", type=float, default=0.03, help="inverse L1 strength; 0.03 keeps ~100 features")
    p.add_argument("--threshold", type=float, default=None, help="override the KatFish Youden threshold")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "heads", "ko.json"))
    args = p.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    X, y = katfish_features(args.katfish, args.cache)

    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        oof[te] = fit(X[tr], y[tr], args.C).decision_function(X[te])
    fpr, tpr, thr = roc_curve(y, oof)
    best = np.argmax(tpr - fpr)
    threshold = float(thr[best]) if args.threshold is None else args.threshold

    clf = fit(X, y, args.C)
    w = clf.coef_[0]
    ids = np.flatnonzero(w)
    head = {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "transform": "log1p",
        "bias": float(clf.intercept_[0]),
        "threshold": threshold,
        "trained_on": f"KatFish {'+'.join(GENRES)} ({len(y)} texts, {int(y.sum())} AI), L1 logreg C={args.C}",
        "features": [{"id": int(i), "weight": float(w[i]), "description": None} for i in ids[np.argsort(-np.abs(w[ids]))]],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(head, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(ids)} features, threshold {threshold:.3f} (KatFish Youden {thr[best]:.3f}: OOF TPR {tpr[best]:.2f} FPR {fpr[best]:.2f}) -> {args.out}")


if __name__ == "__main__":
    main()
