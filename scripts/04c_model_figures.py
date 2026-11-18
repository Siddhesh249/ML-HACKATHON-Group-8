#!/usr/bin/env python3
"""Stage 4c — report figures for §3–4 (T30), generated only from real results.

1. ``results/figures/model_comparison_bars.png`` — CV vs student-test Top-1,
   macro F1 and log loss per model (from ``results/models/cv_vs_student_test.csv``).
2. ``results/figures/learning_curves.png`` — genuine per-iteration histories
   for the only two models that expose them: HistGradientBoosting
   (``train_score_`` / ``validation_score_``, binary log loss) and the MLP
   (``loss_curve_`` / ``validation_scores_``). Both are fit on the training
   folds of CV fold 0 (train trips only). Tree ensembles, NB, LR and the SVM
   have no epoch history, so no curve is drawn for them.
"""

from __future__ import annotations

import argparse
import json
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

LABELS = {
    "hist_gradient_boosting": "HistGB", "random_forest": "Random forest", "extra_trees": "Extra Trees",
    "decision_tree": "Decision tree", "mlp": "MLP", "linear_svm": "Linear SVM",
    "logistic_regression": "Logistic reg.", "gaussian_nb": "Gaussian NB",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(config.CACHE_DIR / "dataset.npz"))
    parser.add_argument("--models-dir", default=str(config.RESULTS_DIR / "models"))
    parser.add_argument("--best-params", default=str(config.RESULTS_DIR / "optuna" / "best_params.json"))
    parser.add_argument("--figures-dir", default=str(config.RESULTS_DIR / "figures"))
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def bars(models_dir: Path, out: Path) -> None:
    frame = pd.read_csv(models_dir / "cv_vs_student_test.csv")
    frame = frame[frame["model"].isin(LABELS)].sort_values("cv_top1", ascending=False)
    names = [LABELS[m] for m in frame["model"]]
    x = np.arange(len(frame))
    w = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (cv_col, st_col, title) in zip(axes, [
        ("cv_top1", "student_top1", "Observation Top-1"),
        ("cv_macro_f1", "student_macro_f1", "Candidate macro F1"),
        ("cv_log_loss", "student_log_loss", "Log loss (lower is better, log scale)"),
    ]):
        ax.bar(x - w / 2, frame[cv_col], w, label="5-fold CV (80 train trips)", color="#3b6ea8")
        ax.bar(x + w / 2, frame[st_col], w, label="student-test (10 trips)", color="#c07a2c")
        ax.set_xticks(x, names, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        if "log" in cv_col:
            ax.set_yscale("log")
        elif "top1" in cv_col:
            ax.set_ylim(0.8, 0.95)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Model comparison — untuned defaults (results/models/cv_vs_student_test.csv)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "model_comparison_bars.png", dpi=150)
    plt.close(fig)


def learning_curves(args, out: Path) -> dict:
    data = dataset.load_dataset(args.dataset)
    X, y, groups = data["X"], data["y"], data["row_trip"]
    split = dataset.cv_split_table(data["trip_ids"], groups, n_splits=5)
    test_trips = [r["trip_index"] for r in split if r["fold"] == 0]
    train_idx = np.flatnonzero(~np.isin(groups, test_trips) & (y >= 0))
    keep = models.subsample_negatives(y[train_idx], ratio=3.0, seed=args.seed)
    train_idx = train_idx[keep]

    best = {}
    if Path(args.best_params).exists():
        best = json.loads(Path(args.best_params).read_text()).get("best_params", {})
    histories = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for label, params in (("untuned", {}), ("tuned", best)):
        if label == "tuned" and not params:
            continue
        hgb = models.build_estimator("hist_gradient_boosting", params, seed=args.seed)
        hgb.fit(X[train_idx], y[train_idx].astype(int))
        tr, va = -np.asarray(hgb.train_score_), -np.asarray(hgb.validation_score_)
        histories[f"hgb_{label}"] = {"n_iter": int(hgb.n_iter_), "train_loss_final": float(tr[-1]),
                                     "val_loss_final": float(va[-1]), "val_loss_min": float(va.min())}
        style = "-" if label == "tuned" else "--"
        axes[0].plot(tr, style, color="#3b6ea8", label=f"{label}: training loss")
        axes[0].plot(va, style, color="#c07a2c", label=f"{label}: validation loss (early-stopping split)")
    axes[0].set_xlabel("boosting iteration")
    axes[0].set_ylabel("binary log loss")
    axes[0].set_title("HistGradientBoosting — real per-iteration history", fontsize=10)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    mlp = models.build_estimator("mlp", {}, seed=args.seed)
    mlp.fit(X[train_idx], y[train_idx].astype(int))
    net = mlp.named_steps["model"]
    loss, val_acc = np.asarray(net.loss_curve_), np.asarray(net.validation_scores_)
    histories["mlp"] = {"n_epochs": int(net.n_iter_), "train_loss_final": float(loss[-1]),
                        "best_val_accuracy": float(val_acc.max())}
    axes[1].plot(loss, color="#3b6ea8", label="training log loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("training log loss", color="#3b6ea8")
    twin = axes[1].twinx()
    twin.plot(val_acc, color="#c07a2c", label="validation accuracy (early-stopping split)")
    twin.set_ylabel("validation accuracy", color="#c07a2c")
    axes[1].set_title("MLP (64-32, untuned) — real per-epoch history", fontsize=10)
    axes[1].grid(alpha=0.3)
    fig.suptitle("Learning curves — fit on CV fold-0 training trips; validation = sklearn's internal 10% "
                 "early-stopping split (rows, not trips)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=150)
    plt.close(fig)
    return histories


def main() -> int:
    args = parse_args()
    out = Path(args.figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    bars(Path(args.models_dir), out)
    histories = learning_curves(args, out)
    with open(Path(args.models_dir) / "learning_curves.json", "w", encoding="utf-8") as fh:
        json.dump(histories, fh, indent=2)
    print(json.dumps(histories, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
