"""
Benchmark harness for a peptide-GPCR interaction dataset.

The core scientific point: interaction datasets leak. If peptides for the same
receptor appear in both train and test, a model memorises the receptor instead
of learning interaction rules, and the reported AUC is optimistic. This harness
reports BOTH a naive random split and a leakage-safe receptor-grouped split so
the gap is visible -- that gap is the single most important number when you
benchmark a proprietary interaction dataset before trusting any model on it.

Baselines:
  - Majority / prevalence (reference floor)
  - Logistic Regression
  - Random Forest
  - Gradient Boosting

Metrics: AUROC, AUPR (average precision), and precision@k (k = number of true
positives in the test fold), reported as mean +/- SD over CV folds.

Run:
  python -m src.benchmark                    # self-contained physicochemical features
  python -m src.benchmark --features esm     # ESM-2 embeddings (needs fair-esm+torch)
  python -m src.benchmark --data yourfile.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .featurize import esm_available, featurize_pairs, featurize_pairs_esm
from .make_dataset import make

MODELS = {
    "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
    "GradBoost": lambda: GradientBoostingClassifier(random_state=0),
}


def precision_at_k(y_true, y_score):
    k = int(np.sum(y_true))
    if k == 0:
        return np.nan
    idx = np.argsort(y_score)[::-1][:k]
    return float(np.mean(np.asarray(y_true)[idx]))


def _eval(splitter, X, y, groups=None):
    """Return {model: {metric: (mean, sd)}} over folds for one split strategy."""
    out = {m: {"auroc": [], "aupr": [], "p@k": []} for m in MODELS}
    out["Prevalence"] = {"auroc": [], "aupr": [], "p@k": []}
    split_iter = (splitter.split(X, y, groups) if groups is not None
                  else splitter.split(X, y))
    for tr, te in split_iter:
        prior = float(np.mean(y[tr]))
        # prevalence baseline: constant score = train positive rate
        out["Prevalence"]["auroc"].append(0.5)
        out["Prevalence"]["aupr"].append(average_precision_score(y[te], np.full(len(te), prior)))
        out["Prevalence"]["p@k"].append(precision_at_k(y[te], np.random.default_rng(0).random(len(te))))
        for name, ctor in MODELS.items():
            clf = ctor()
            clf.fit(X[tr], y[tr])
            score = clf.predict_proba(X[te])[:, 1]
            out[name]["auroc"].append(roc_auc_score(y[te], score))
            out[name]["aupr"].append(average_precision_score(y[te], score))
            out[name]["p@k"].append(precision_at_k(y[te], score))
    return {m: {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in md.items()}
            for m, md in out.items()}


def run(df: pd.DataFrame, features: str = "physchem", n_splits: int = 5):
    y = df["label"].to_numpy()
    groups = df["receptor_id"].astype("category").cat.codes.to_numpy()
    if features == "esm":
        if not esm_available():
            raise SystemExit("ESM requested but fair-esm/torch not installed. Use --features physchem.")
        X = featurize_pairs_esm(df["peptide_seq"].tolist(), df["receptor_seq"].tolist())
    else:
        X = featurize_pairs(df["peptide_seq"].tolist(), df["receptor_seq"].tolist())

    random_res = _eval(StratifiedKFold(n_splits, shuffle=True, random_state=0), X, y)
    grouped_res = _eval(GroupKFold(n_splits), X, y, groups=groups)
    return random_res, grouped_res, X.shape


def _fmt(res):
    lines = [f"  {'model':<14}{'AUROC':>16}{'AUPR':>16}{'P@k':>14}"]
    for m, md in res.items():
        a, ap, pk = md["auroc"], md["aupr"], md["p@k"]
        lines.append(f"  {m:<14}{a[0]:>7.3f}±{a[1]:<7.3f}{ap[0]:>7.3f}±{ap[1]:<7.3f}"
                     f"{pk[0]:>6.3f}±{pk[1]:<6.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="CSV with peptide_seq,receptor_seq,receptor_id,label")
    ap.add_argument("--features", default="physchem", choices=["physchem", "esm"])
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    if args.data and os.path.exists(args.data):
        df = pd.read_csv(args.data)
        src = args.data
    else:
        df = make()
        src = "synthetic demo (src/make_dataset.make)"

    print(f"Dataset: {src}")
    print(f"  {len(df)} pairs | {df.receptor_id.nunique()} receptors | "
          f"positive rate {df.label.mean():.1%} | features={args.features}")
    random_res, grouped_res, shape = run(df, args.features, args.splits)
    print(f"  feature matrix: {shape[0]} x {shape[1]}\n")

    print("[1] NAIVE random split (optimistic -- peptides of the same receptor leak across folds)")
    print(_fmt(random_res))
    print("\n[2] LEAKAGE-SAFE receptor-grouped split (honest generalisation to unseen receptors)")
    print(_fmt(grouped_res))

    def gap(m):
        return random_res[m]["auroc"][0] - grouped_res[m]["auroc"][0]

    best = max(MODELS, key=lambda m: grouped_res[m]["auroc"][0])
    worst = max(MODELS, key=gap)
    print(f"\nMost robust model (highest grouped AUROC): {best} "
          f"-> {grouped_res[best]['auroc'][0]:.3f} (honest generalisation to unseen receptors)")
    print(f"Largest optimism gap: {worst} random {random_res[worst]['auroc'][0]:.3f} "
          f"-> grouped {grouped_res[worst]['auroc'][0]:.3f} (Δ={gap(worst):+.3f})")
    print("A large gap means the dataset rewards receptor memorisation, not interaction "
          "learning -- report the grouped number, never the random one. This gap is the "
          "single most important output of benchmarking a proprietary interaction dataset.")


if __name__ == "__main__":
    main()
