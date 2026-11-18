#!/usr/bin/env python3
"""Stage 3 — train and compare at least five model families with grouped CV."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import config, dataset, metrics, models  # noqa: E402

DEFAULT_MODELS = [
    "logistic_regression",
    "gaussian_nb",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
]

FAST_PARAMS = {
    "random_forest": {"n_estimators": 120, "max_depth": 20, "min_samples_leaf": 10},
    "extra_trees": {"n_estimators": 120, "max_depth": 20, "min_samples_leaf": 10},
    "hist_gradient_boosting": {"max_iter": 200, "learning_rate": 0.1},
    "mlp": {"max_iter": 120},
    "logistic_regression": {"max_iter": 1000},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(config.CACHE_DIR / "dataset.npz"))
    parser.add_argument("--out-dir", default=str(config.RESULTS_DIR / "models"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--max-models", type=int, default=None, help="Debug: run only the first N models")
    parser.add_argument(
        "--profile",
        choices=["full", "fast"],
        default="full",
        help="fast reduces ensemble size for an integration pass; report the profile used",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = dataset.load_dataset(args.dataset)
    X, y = data["X"], data["y"]
    groups = data["row_trip"]
    obs_offsets = data["obs_offsets"]
    row_arc = data["row_arc"]
    labels = data["labels"]
    confidence = data["confidence"]
    if X.shape[0] == 0:
        print("Dataset is empty. Run scripts/02_build_dataset.py first.")
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.dataset).parent
    model_names = args.models[: args.max_models] if args.max_models else args.models
    rows = []

    for name in model_names:
        print(f"\n=== {name} ({models.MODEL_FAMILIES.get(name, 'unknown')}) ===", flush=True)
        params = FAST_PARAMS.get(name, {}) if args.profile == "fast" else {}
        start = time.perf_counter()
        oof, folds, train_seconds = models.cross_validate_oof(
            X,
            y,
            groups,
            name,
            params=params,
            n_splits=args.folds,
            negative_ratio=args.negative_ratio,
            seed=args.seed,
        )
        total_seconds = time.perf_counter() - start
        candidate = metrics.candidate_metrics(y, oof)
        top1, n_eval, n_correct = dataset.top1_from_probs(
            oof, obs_offsets, row_arc, labels, confidence
        )
        # Measure pure inference throughput on a deterministic sample.
        estimator = models.build_estimator(name, params, seed=args.seed)
        fit_mask = y >= 0
        fit_n = min(200_000, int(fit_mask.sum()))
        estimator.fit(X[fit_mask][:fit_n], y[fit_mask][:fit_n].astype(int))
        sample = X[np.linspace(0, X.shape[0] - 1, min(20_000, X.shape[0])).astype(int)]
        t0 = time.perf_counter()
        estimator.predict_proba(sample)
        inference_ms = (time.perf_counter() - t0) * 1000.0 / max(sample.shape[0], 1)

        row = {
            "model": name,
            "family": models.MODEL_FAMILIES.get(name, "unknown"),
            "config": f"{args.profile} profile: {params or 'library defaults'}",
            "profile": args.profile,
            **candidate,
            "top1_accuracy": float(top1),
            "top1_n": int(n_eval),
            "top1_correct": int(n_correct),
            "train_seconds": float(train_seconds),
            "cv_seconds": float(total_seconds),
            "inference_ms_per_candidate": float(inference_ms),
            "n_folds": len(folds),
        }
        rows.append(row)
        np.save(cache / f"oof_{name}.npy", oof)
        with open(out / f"{name}_folds.json", "w", encoding="utf-8") as fh:
            json.dump(folds, fh, indent=2)
        print(
            f"top1={top1:.4f} macroF1={candidate.get('macro_f1', float('nan')):.4f} "
            f"logloss={candidate.get('log_loss', float('nan')):.4f} "
            f"train={train_seconds:.1f}s",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["top1_accuracy", "macro_f1"], ascending=False)
    csv_path = out / "model_comparison.csv"
    frame.to_csv(csv_path, index=False)

    best = frame.iloc[0].to_dict()
    selected = {
        "model": best["model"],
        "family": best["family"],
        "params": {},
        "cv_top1": float(best["top1_accuracy"]),
        "cv_macro_f1": float(best["macro_f1"]),
        "selected_by": "CV observation Top-1, tie-break macro F1",
        "dataset": str(Path(args.dataset).resolve()),
        "seed": args.seed,
        "folds": args.folds,
        "profile": args.profile,
    }
    with open(out / "selected_model.json", "w", encoding="utf-8") as fh:
        json.dump(selected, fh, indent=2)

    print("\n" + frame[["model", "family", "top1_accuracy", "macro_f1", "f1", "log_loss", "train_seconds"]].to_string(index=False))
    print(f"\nSelected: {best['model']} -> {out / 'selected_model.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
