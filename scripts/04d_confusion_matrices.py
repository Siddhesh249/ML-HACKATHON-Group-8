#!/usr/bin/env python3
"""Candidate-level confusion matrices on student-test for every model (report §3.3).

Each model is refit on all 80 train trips with the same negative subsampling as
the comparison and the exported artifact, then thresholded at 0.5 on every
student-test candidate pair. Writes ``results/models/confusion_matrices.csv``
and ``results/figures/confusion_matrices.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import config, dataset, models  # noqa: E402

MODELS = [
    ("hist_gradient_boosting", "HistGradientBoosting"), ("random_forest", "Random forest"),
    ("extra_trees", "Extra Trees"), ("decision_tree", "Decision tree"), ("mlp", "MLP"),
    ("linear_svm", "Linear SVM"), ("logistic_regression", "Logistic regression"), ("gaussian_nb", "Gaussian NB"),
]


def main() -> int:
    train = dataset.load_dataset(config.CACHE_DIR / "dataset.npz")
    test = dataset.load_dataset(config.CACHE_DIR / "student_test.npz")
    y_tr, y_te = train["y"], test["y"]
    fit_idx = np.flatnonzero(y_tr >= 0)
    fit_idx = fit_idx[models.subsample_negatives(y_tr[fit_idx], ratio=3.0, seed=config.SEED)]
    mask = y_te >= 0
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for ax, (name, label) in zip(axes.ravel(), MODELS):
        est = models.build_estimator(name, {}, seed=config.SEED)
        est.fit(train["X"][fit_idx], y_tr[fit_idx].astype(int))
        pred = (est.predict_proba(test["X"][mask])[:, 1] >= 0.5).astype(int)
        truth = y_te[mask].astype(int)
        tn = int(((pred == 0) & (truth == 0)).sum()); fp = int(((pred == 1) & (truth == 0)).sum())
        fn = int(((pred == 0) & (truth == 1)).sum()); tp = int(((pred == 1) & (truth == 1)).sum())
        rows.append({"model": name, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                     "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)})
        cm = np.array([[tn, fp], [fn, tp]], dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j]):,}\n({norm[i, j]:.1%})", ha="center", va="center",
                        color="white" if norm[i, j] > 0.6 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["pred: wrong arc", "pred: true arc"], fontsize=8)
        ax.set_yticks([0, 1], ["wrong arc", "true arc"], fontsize=8)
        ax.set_title(label, fontsize=10)
        print(f"{label:<22} tp={tp} fp={fp} fn={fn} tn={tn}", flush=True)
    fig.suptitle("Candidate-level confusion matrices on student-test (threshold 0.5; rows = truth, row-normalised)",
                 fontsize=11)
    fig.tight_layout()
    out_fig = config.RESULTS_DIR / "figures" / "confusion_matrices.png"
    fig.savefig(out_fig, dpi=140)
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "models" / "confusion_matrices.csv", index=False)
    print(f"Wrote {out_fig.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
