#!/usr/bin/env python3
"""Stage 4 — Bayesian hyperparameter optimisation with Optuna (TPE).

Group-aware CV over trips. The evaluator partition is never used here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import config, dataset, models  # noqa: E402

SPACES = {
    "hist_gradient_boosting": {
        "learning_rate": ("float", 0.01, 0.3, True),
        "max_leaf_nodes": ("int", 15, 127),
        "min_samples_leaf": ("int", 20, 200),
        "l2_regularization": ("float", 1e-8, 10.0, True),
        "max_iter": ("int", 100, 600),
    },
    "random_forest": {
        "n_estimators": ("int", 150, 600),
        "max_depth": ("int", 6, 40),
        "min_samples_leaf": ("int", 1, 40),
        "max_features": ("categorical", ["sqrt", "log2", 0.3]),
    },
    "extra_trees": {
        "n_estimators": ("int", 150, 600),
        "max_depth": ("int", 6, 40),
        "min_samples_leaf": ("int", 1, 40),
        "max_features": ("categorical", ["sqrt", "log2", 0.3]),
    },
    "logistic_regression": {
        "C": ("float", 1e-3, 100.0, True),
        "max_iter": ("int", 500, 3000),
    },
    "decision_tree": {
        "max_depth": ("int", 4, 40),
        "min_samples_leaf": ("int", 5, 200),
    },
    "mlp": {
        "alpha": ("float", 1e-6, 1e-1, True),
        "learning_rate_init": ("float", 1e-4, 1e-2, True),
        "max_iter": ("int", 100, 400),
    },
    "linear_svm": {
        "C": ("float", 1e-3, 100.0, True),
        "calibration_cv": ("int", 2, 4),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(config.CACHE_DIR / "dataset.npz"))
    parser.add_argument("--selected", default=str(config.RESULTS_DIR / "models" / "selected_model.json"))
    parser.add_argument("--model", default=None, help="Override the selected model name")
    parser.add_argument("--out-dir", default=str(config.RESULTS_DIR / "optuna"))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=None, help="Seconds; optional")
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import optuna
    except ImportError:
        print("Optuna is not installed. Run: pip install optuna")
        return 1

    data = dataset.load_dataset(args.dataset)
    X, y, groups = data["X"], data["y"], data["row_trip"]
    obs_offsets, row_arc = data["obs_offsets"], data["row_arc"]
    labels, confidence = data["labels"], data["confidence"]

    model_name = args.model
    params = {}
    if model_name is None:
        with open(args.selected, encoding="utf-8") as fh:
            selected = json.load(fh)
        model_name = selected["model"]
        params = selected.get("params", {})
    if model_name not in SPACES:
        print(f"No search space defined for {model_name}. Add it to SPACES.")
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{out / 'study.db'}"

    def suggest(trial):
        suggested = {}
        for key, spec in SPACES[model_name].items():
            if spec[0] == "float":
                _, low, high, log = spec
                suggested[key] = trial.suggest_float(key, low, high, log=log)
            elif spec[0] == "int":
                _, low, high = spec
                suggested[key] = trial.suggest_int(key, low, high)
            else:
                _, choices = spec
                suggested[key] = trial.suggest_categorical(key, choices)
        return suggested

    def objective(trial):
        trial_params = suggest(trial)
        oof, _, _ = models.cross_validate_oof(
            X,
            y,
            groups,
            model_name,
            params=trial_params,
            n_splits=args.folds,
            negative_ratio=3.0,
            seed=args.seed,
        )
        top1, _, _ = dataset.top1_from_probs(oof, obs_offsets, row_arc, labels, confidence)
        return float(top1) if math.isfinite(top1) else 0.0

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=f"pathsnap_{model_name}",
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )
    print(f"Optimising {model_name}: {args.trials} trials × {args.folds}-fold GroupKFold")
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, show_progress_bar=False)

    trials = study.trials_dataframe()
    trials.to_csv(out / "trials.csv", index=False)
    best = {
        "model": model_name,
        "objective": "mean CV observation Top-1",
        "folds": args.folds,
        "n_trials": int(len(study.trials)),
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "study": str(out / "study.db"),
        "seed": args.seed,
    }
    with open(out / "best_params.json", "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=2)

    # Plots: real Optuna history, not fabricated curves.
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
        )

        ax = plot_optimization_history(study)
        ax.figure.tight_layout()
        ax.figure.savefig(out / "optimization_history.png", dpi=140)
        if len(study.trials) >= 4:
            ax = plot_param_importances(study)
            ax.figure.tight_layout()
            ax.figure.savefig(out / "param_importances.png", dpi=140)
    except Exception as exc:  # noqa: BLE001 - plots are optional
        print(f"Plot generation skipped: {exc}")

    print(json.dumps(best, indent=2))
    print(f"Trials written to {out / 'trials.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
