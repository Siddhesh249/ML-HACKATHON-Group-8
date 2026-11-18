#!/usr/bin/env python3
"""Stage 3 — train and compare at least five model families with grouped CV.

For every model this records, per GroupKFold fold (groups = trips):
validation observation Top-1, candidate macro/positive F1 and log loss, the
same metrics on the training-fold trips (over/underfitting evidence), fit time
and inference time. A rule-based nearest-road baseline is reported alongside
but is never selectable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
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
    "mlp",
    "linear_svm",
]

FAST_PARAMS = {
    "random_forest": {"n_estimators": 120, "max_depth": 20, "min_samples_leaf": 10},
    "extra_trees": {"n_estimators": 120, "max_depth": 20, "min_samples_leaf": 10},
    "hist_gradient_boosting": {"max_iter": 200, "learning_rate": 0.1},
    "mlp": {"max_iter": 120},
    "logistic_regression": {"max_iter": 1000},
}

# Effective hyperparameters of the full profile, written to the CSV so the
# report can quote them without reading code (mirrors models.build_estimator).
FULL_CONFIG = {
    "logistic_regression": "StandardScaler + L2 LR, C=1.0, class_weight=balanced, lbfgs, max_iter=2000",
    "gaussian_nb": "var_smoothing=1e-9",
    "decision_tree": "max_depth=12, min_samples_leaf=20, class_weight=balanced",
    "random_forest": "300 trees, max_depth=None, min_samples_leaf=5, max_features=sqrt, class_weight=balanced_subsample",
    "extra_trees": "300 trees, max_depth=None, min_samples_leaf=5, max_features=sqrt, class_weight=balanced",
    "hist_gradient_boosting": "lr=0.08, max_iter=300, max_leaf_nodes=31, min_samples_leaf=40, l2=1.0, early_stopping (10% val)",
    "mlp": "StandardScaler + MLP(64,32) relu, adam lr=1e-3, alpha=1e-4, max_iter=200, early_stopping",
    "linear_svm": "StandardScaler + LinearSVC C=1.0 class_weight=balanced, sigmoid-calibrated (cv=3)",
}

NOT_SELECTABLE = {"nearest_road_baseline"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(config.CACHE_DIR / "dataset.npz"))
    parser.add_argument("--out-dir", default=str(config.RESULTS_DIR / "models"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--max-models", type=int, default=None, help="Debug: run only the first N models")
    parser.add_argument("--no-train-scores", action="store_true", help="Skip training-side metrics (faster)")
    parser.add_argument(
        "--profile",
        choices=["full", "fast"],
        default="full",
        help="fast reduces ensemble size for an integration pass; report the profile used",
    )
    return parser.parse_args()


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def make_fold_scorer(data: dict, with_train: bool, timing: dict):
    X, y = data["X"], data["y"]
    row_obs, row_arc = data["row_obs"], data["row_arc"]
    labels, confidence = data["labels"], data["confidence"]
    obs_trip, row_trip = data["obs_trip"], data["row_trip"]
    n_rows = y.size

    def top1_on(rows: np.ndarray, prob: np.ndarray) -> float:
        full = np.full(n_rows, np.nan)
        full[rows] = prob
        trips = np.unique(row_trip[rows])
        top1, _, _ = dataset.top1_by_mask(
            full, row_obs, row_arc, labels, confidence, np.isin(obs_trip, trips)
        )
        return top1

    def scorer(fold, estimator, train_idx, test_idx, test_prob):
        out = {}
        val = metrics.candidate_metrics(y[test_idx], test_prob)
        out["val_top1"] = top1_on(test_idx, test_prob)
        out["val_macro_f1"] = val.get("macro_f1", float("nan"))
        out["val_f1"] = val.get("f1", float("nan"))
        out["val_log_loss"] = val.get("log_loss", float("nan"))
        if with_train:
            train_prob = estimator.predict_proba(X[train_idx])[:, 1]
            tr = metrics.candidate_metrics(y[train_idx], train_prob)
            out["train_top1"] = top1_on(train_idx, train_prob)
            out["train_macro_f1"] = tr.get("macro_f1", float("nan"))
            out["train_log_loss"] = tr.get("log_loss", float("nan"))
        # Inference throughput on held-out rows, same estimator as the fold.
        sample = test_idx[np.linspace(0, test_idx.size - 1, min(20_000, test_idx.size)).astype(int)]
        t0 = time.perf_counter()
        estimator.predict_proba(X[sample])
        out["inference_ms_per_candidate"] = (time.perf_counter() - t0) * 1000.0 / max(sample.size, 1)
        timing.setdefault("inference", []).append(out["inference_ms_per_candidate"])
        return out

    return scorer


def nearest_road_baseline(data: dict, folds: int, run_meta: dict) -> dict:
    """Argmin perpendicular distance per observation (no learning)."""
    X = data["X"]
    dist_col = list(config.FEATURE_NAMES).index("log_distance")
    score = -X[:, dist_col].astype(float)  # higher = closer; only the argmax matters
    top1, n_eval, n_correct = dataset.top1_from_probs(
        score, data["obs_offsets"], data["row_arc"], data["labels"], data["confidence"]
    )
    fold_top1 = []
    split = pd.DataFrame(dataset.cv_split_table(data["trip_ids"], data["row_trip"], folds))
    for fold in sorted(split["fold"].unique()):
        trips = split.loc[split["fold"] == fold, "trip_index"].to_numpy()
        t, _, _ = dataset.top1_by_mask(
            score, data["row_obs"], data["row_arc"], data["labels"], data["confidence"],
            np.isin(data["obs_trip"], trips),
        )
        fold_top1.append(t)
    mean, std = _mean_std(fold_top1)
    return {
        **run_meta,
        "model": "nearest_road_baseline",
        "family": "rule-based (reference only)",
        "config": "argmin perpendicular distance; no fitting; no probabilities",
        "top1_accuracy": float(top1),
        "top1_fold_mean": mean,
        "top1_fold_std": std,
        "top1_n": int(n_eval),
        "top1_correct": int(n_correct),
        "n_folds": len(fold_top1),
        "notes": "reference row; not selectable; candidate metrics undefined (no probabilities)",
    }


def main() -> int:
    args = parse_args()
    data = dataset.load_dataset(args.dataset)
    X, y = data["X"], data["y"]
    groups = data["row_trip"]
    if X.shape[0] == 0:
        print("Dataset is empty. Run scripts/02_build_dataset.py first.")
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.dataset).parent
    model_names = args.models[: args.max_models] if args.max_models else args.models
    commit = models.git_commit()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_meta = {
        "run_id": run_id,
        "git_commit": commit,
        "seed": args.seed,
        "partition": "train (GroupKFold by trip)",
        "profile": args.profile,
        "negative_ratio": args.negative_ratio,
    }
    rows = [nearest_road_baseline(data, args.folds, run_meta)]
    print(f"nearest_road_baseline top1={rows[0]['top1_accuracy']:.4f}", flush=True)

    for name in model_names:
        print(f"\n=== {name} ({models.MODEL_FAMILIES.get(name, 'unknown')}) ===", flush=True)
        params = FAST_PARAMS.get(name, {}) if args.profile == "fast" else {}
        timing: dict = {}
        scorer = make_fold_scorer(data, with_train=not args.no_train_scores, timing=timing)
        start = time.perf_counter()
        try:
            oof, folds, train_seconds = models.cross_validate_oof(
                X,
                y,
                groups,
                name,
                params=params,
                n_splits=args.folds,
                negative_ratio=args.negative_ratio,
                seed=args.seed,
                fold_scorer=scorer,
            )
        except Exception as exc:  # keep failed runs in the record
            print(f"FAILED: {exc!r}", flush=True)
            rows.append({**run_meta, "model": name, "family": models.MODEL_FAMILIES.get(name, "unknown"),
                         "status": "failed", "notes": repr(exc)})
            continue
        total_seconds = time.perf_counter() - start
        candidate = metrics.candidate_metrics(y, oof)
        top1, n_eval, n_correct = dataset.top1_from_probs(
            oof, data["obs_offsets"], data["row_arc"], data["labels"], data["confidence"]
        )

        def agg(key: str) -> tuple[float, float]:
            return _mean_std([f.get(key, float("nan")) for f in folds])

        top1_m, top1_s = agg("val_top1")
        mf1_m, mf1_s = agg("val_macro_f1")
        ll_m, ll_s = agg("val_log_loss")
        tr_top1_m, _ = agg("train_top1")
        tr_mf1_m, _ = agg("train_macro_f1")
        tr_ll_m, _ = agg("train_log_loss")
        inf_m, _ = agg("inference_ms_per_candidate")

        config_text = (
            f"{args.profile} profile: "
            + (json.dumps(params) if params else FULL_CONFIG.get(name, "library defaults"))
        )
        row = {
            **run_meta,
            "model": name,
            "family": models.MODEL_FAMILIES.get(name, "unknown"),
            "config": config_text,
            **candidate,
            "top1_accuracy": float(top1),
            "top1_fold_mean": top1_m,
            "top1_fold_std": top1_s,
            "macro_f1_fold_mean": mf1_m,
            "macro_f1_fold_std": mf1_s,
            "log_loss_fold_mean": ll_m,
            "log_loss_fold_std": ll_s,
            "train_top1_fold_mean": tr_top1_m,
            "train_macro_f1_fold_mean": tr_mf1_m,
            "train_log_loss_fold_mean": tr_ll_m,
            "top1_train_minus_val": tr_top1_m - top1_m,
            "top1_n": int(n_eval),
            "top1_correct": int(n_correct),
            "train_seconds": float(train_seconds),
            "cv_seconds": float(total_seconds),
            "inference_ms_per_candidate": inf_m,
            "n_folds": len(folds),
            "status": "ok",
            "notes": "",
        }
        rows.append(row)
        np.save(cache / f"oof_{name}.npy", oof)
        with open(out / f"{name}_folds.json", "w", encoding="utf-8") as fh:
            json.dump({"run_id": run_id, "model": name, "config": config_text, "folds": folds}, fh, indent=2)
        print(
            f"top1={top1:.4f} (fold {top1_m:.4f}±{top1_s:.4f}, train {tr_top1_m:.4f}) "
            f"macroF1={candidate.get('macro_f1', float('nan')):.4f} "
            f"logloss={candidate.get('log_loss', float('nan')):.4f} "
            f"train={train_seconds:.1f}s",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["top1_accuracy", "macro_f1"], ascending=False, na_position="last")
    csv_path = out / "model_comparison.csv"
    frame.to_csv(csv_path, index=False)

    eligible = frame[(~frame["model"].isin(NOT_SELECTABLE)) & (frame.get("status", "ok") == "ok")]
    best = eligible.iloc[0].to_dict()
    selected = {
        "model": best["model"],
        "family": best["family"],
        "params": FAST_PARAMS.get(best["model"], {}) if args.profile == "fast" else {},
        "config": best["config"],
        "cv_top1": float(best["top1_accuracy"]),
        "cv_top1_fold_mean": float(best["top1_fold_mean"]),
        "cv_top1_fold_std": float(best["top1_fold_std"]),
        "cv_macro_f1": float(best["macro_f1"]),
        "cv_f1": float(best["f1"]),
        "cv_log_loss": float(best["log_loss"]),
        "selected_by": "CV observation Top-1 (pooled OOF), tie-break macro F1; guardrails: positive F1, log loss",
        "status": "preliminary (Phase 1, untuned); Optuna (T23) may change params",
        "runner_up": eligible.iloc[1]["model"] if len(eligible) > 1 else None,
        "dataset": models.repo_relative(args.dataset),
        "run_id": run_id,
        "git_commit": commit,
        "seed": args.seed,
        "folds": args.folds,
        "profile": args.profile,
        "negative_ratio": args.negative_ratio,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(["python3", "scripts/03_train_models.py", *sys.argv[1:]]),
    }
    with open(out / "selected_model.json", "w", encoding="utf-8") as fh:
        json.dump(selected, fh, indent=2)

    cols = ["model", "family", "top1_accuracy", "top1_fold_std", "train_top1_fold_mean",
            "macro_f1", "f1", "log_loss", "train_seconds", "inference_ms_per_candidate"]
    print("\n" + frame[[c for c in cols if c in frame.columns]].to_string(index=False))
    print(f"\nSelected: {best['model']} -> {out / 'selected_model.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
