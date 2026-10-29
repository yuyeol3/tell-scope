"""Trains the Korean head over the open-domain detector's SAE features.

The released DetectorScope heads are trained on English and score Korean at
chance, but the SAE features underneath still separate Korean human and AI
text. This fits a sparse L1 logistic regression on log1p of the max-pooled
features of KatFish (https://github.com/Shinwoo-Park/katfishnet, essays and
abstracts; poetry hurt transfer to prose) and writes heads/ko.json. Feature
descriptions are added afterwards from scripts/ko_snippets.py (see README).

  python scripts/train_ko_head.py --katfish path/to/katfish_dataset --cache work/ \\
      --max-layer 18 --threshold 0.2 [--reuse old_head.json]

The released head was trained with exactly that command (65 features, all
with descriptions carried over via --reuse). Using SAEs up to layer 18 only
matched or beat all four layers (AUROC 0.97 Namuwiki/blog, 0.87 PR pairs) and
lets ko_model.py drop the deeper blocks.

The decision threshold maximizes Youden's J on 5-fold out-of-fold scores
unless --threshold is given. KatFish's 2024-era AI texts score far higher than
current models', so its own threshold (+0.89) let most Claude-written blog
and PR texts pass as human. The released head uses +0.2, Youden's J on the PR
pairs; on the Namuwiki/blog pairs it gives TPR 0.82 FPR 0.03. Those pairs are
not redistributable, so they are not in this repository.
"""

import argparse
import json
import os

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

from score import BASE_MODEL, BASE_REVISION, D_SAE, HOOK_LAYERS, load, plain

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
    p.add_argument("--max-layer", type=int, default=max(HOOK_LAYERS),
                   help="use only SAEs at hooks <= this layer, so the model can be cut there")
    p.add_argument("--reuse", default=None, help="head JSON whose descriptions to carry over for shared feature ids")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "heads", "ko.json"))
    args = p.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    X, y = katfish_features(args.katfish, args.cache)
    keep = [i for i, layer in enumerate(HOOK_LAYERS) if layer <= args.max_layer]
    X = X[:, np.concatenate([np.arange(i * D_SAE, (i + 1) * D_SAE) for i in keep])]
    offset = np.concatenate([np.arange(i * D_SAE, (i + 1) * D_SAE) for i in keep])  # column -> global feature id

    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        oof[te] = fit(X[tr], y[tr], args.C).decision_function(X[te])
    fpr, tpr, thr = roc_curve(y, oof)
    best = np.argmax(tpr - fpr)
    threshold = float(thr[best]) if args.threshold is None else args.threshold

    clf = fit(X, y, args.C)
    w = clf.coef_[0]
    cols = np.flatnonzero(w)
    # Features that fire on every document at a near-constant value (the
    # first-token position) carry no style signal; fold them into the bias so
    # they never show up as evidence.
    bias = float(clf.intercept_[0])
    for c in list(cols):
        v = X[:, c].toarray().ravel()
        if (v > 0).mean() > 0.99 and v.std() < 0.01 * v.mean():
            bias += float(w[c] * v.mean())
            cols = cols[cols != c]
    old = {f["id"]: f for f in json.load(open(args.reuse, encoding="utf-8"))["features"]} if args.reuse else {}
    features = []
    for c in cols[np.argsort(-np.abs(w[cols]))]:
        fid = int(offset[c])
        prev = {k: v for k, v in old.get(fid, {}).items() if k not in ("id", "weight")}
        features.append({"id": fid, "weight": float(w[c]), "description": None, **prev})
    head = {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "transform": "log1p",
        "max_layer": max(HOOK_LAYERS[i] for i in keep),
        "bias": bias,
        "threshold": threshold,
        "trained_on": f"KatFish {'+'.join(GENRES)} ({len(y)} texts, {int(y.sum())} AI), L1 logreg C={args.C}",
        "features": features,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(head, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(features)} features ({sum(f['description'] is not None for f in features)} with reused descriptions), threshold {threshold:.3f} (KatFish Youden {thr[best]:.3f}: OOF TPR {tpr[best]:.2f} FPR {fpr[best]:.2f}) -> {args.out}")


if __name__ == "__main__":
    main()
