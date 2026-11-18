#!/usr/bin/env python3
"""Stage 4 — Bayesian hyperparameter optimisation with Optuna (TPE), T23.

Objective: CV observation Top-1 (pooled out-of-fold, the same selection metric
as ``results/models/model_comparison.csv``) with GroupKFold over the 80 train
trips. Trial 0 is the untuned default configuration, so the study itself
records the tuned-vs-untuned gap on identical folds. Every trial also records
macro F1, positive F1, log loss, fold mean/std of Top-1 and fit time.

The student-test and evaluator partitions are never read here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import config, dataset, metrics, models  # noqa: E402

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

# Untuned defaults (mirror models.build_estimator), enqueued as trial 0.
DEFAULTS = {
    "hist_gradient_boosting": {"learning_rate": 0.08, "max_leaf_nodes": 31, "min_samples_leaf": 40,
                               "l2_regularization": 1.0, "max_iter": 300},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(config.DATA_ROOT), help="Unused; accepted for `make tune`")
    parser.add_argument("--dataset", default=str(config.CACHE_DIR / "dataset.npz"))
    parser.add_argument("--selected", default=str(config.RESULTS_DIR / "models" / "selected_model.json"))
    parser.add_argument("--model", default=None, help="Override the selected model name")
    parser.add_argument("--out-dir", default=str(config.RESULTS_DIR / "optuna"))
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--negative-ratio", type=float, default=3.0)
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
    row_obs, row_arc = data["row_obs"], data["row_arc"]
    labels, confidence, obs_trip = data["labels"], data["confidence"], data["obs_trip"]

    model_name = args.model
    if model_name is None:
        with open(args.selected, encoding="utf-8") as fh:
            model_name = json.load(fh)["model"]
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
        fold_top1: list[float] = []

        def scorer(fold, est, train_idx, test_idx, prob):
            full = np.full(y.size, np.nan)
            full[test_idx] = prob
            t, _, _ = dataset.top1_by_mask(full, row_obs, row_arc, labels, confidence,
                                           np.isin(obs_trip, np.unique(groups[test_idx])))
            fold_top1.append(t)
            return {}

        t0 = time.perf_counter()
        oof, _, fit_seconds = models.cross_validate_oof(
            X, y, groups, model_name, params=trial_params, n_splits=args.folds,
            negative_ratio=args.negative_ratio, seed=args.seed, fold_scorer=scorer,
        )
        top1, _, _ = dataset.top1_by_mask(oof, row_obs, row_arc, labels, confidence)
        cand = metrics.candidate_metrics(y, oof)
        trial.set_user_attr("macro_f1", cand.get("macro_f1"))
        trial.set_user_attr("f1", cand.get("f1"))
        trial.set_user_attr("log_loss", cand.get("log_loss"))
        trial.set_user_attr("top1_fold_mean", float(np.mean(fold_top1)))
        trial.set_user_attr("top1_fold_std", float(np.std(fold_top1, ddof=1)))
        trial.set_user_attr("fit_seconds", float(fit_seconds))
        trial.set_user_attr("wall_seconds", float(time.perf_counter() - t0))
        print(f"trial {trial.number:>3}: top1={top1:.4f} macroF1={cand.get('macro_f1', float('nan')):.4f} "
              f"logloss={cand.get('log_loss', float('nan')):.4f} wall={time.perf_counter() - t0:.0f}s "
              f"{trial_params}", flush=True)
        return float(top1) if math.isfinite(top1) else 0.0

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=f"pathsnap_{model_name}",
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )
    if model_name in DEFAULTS and not study.trials:
        study.enqueue_trial(DEFAULTS[model_name], user_attrs={"role": "untuned_default"})
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"Optimising {model_name}: {args.trials} trials × {args.folds}-fold GroupKFold (trial 0 = untuned)")
    t_start = time.perf_counter()
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, catch=(Exception,))
    search_seconds = time.perf_counter() - t_start

    trials = study.trials_dataframe()
    trials.to_csv(out / "trials.csv", index=False)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    baseline = next((t for t in complete if t.user_attrs.get("role") == "untuned_default"), None)
    best_trial = study.best_trial
    best = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(["python3", "scripts/04_tune.py", *sys.argv[1:]]),
        "model": model_name,
        "sampler": "TPE (seed %d)" % args.seed,
        "objective": "CV observation Top-1 (pooled out-of-fold), GroupKFold by trip on train",
        "folds": args.folds,
        "negative_ratio": args.negative_ratio,
        "search_space": {k: list(v) for k, v in SPACES[model_name].items()},
        "n_trials": int(len(study.trials)),
        "n_complete": int(len(complete)),
        "n_failed": int(sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials)),
        "search_wall_seconds_this_run": float(search_seconds),
        "search_wall_seconds_all_trials": float(sum(t.user_attrs.get("wall_seconds", 0.0) for t in complete)),
        "best_trial": int(best_trial.number),
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_metrics": {k: best_trial.user_attrs.get(k) for k in
                         ("macro_f1", "f1", "log_loss", "top1_fold_mean", "top1_fold_std", "fit_seconds")},
        "untuned_value": float(baseline.value) if baseline else None,
        "untuned_metrics": {k: baseline.user_attrs.get(k) for k in
                            ("macro_f1", "f1", "log_loss", "top1_fold_mean", "top1_fold_std", "fit_seconds")}
        if baseline else None,
        "study": models.repo_relative(out / "study.db"),
        "seed": args.seed,
    }
    with open(out / "best_params.json", "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=2)

    # Plots: real Optuna history, not fabricated curves.
    try:
        import matplotlib

        matplotlib.use("Agg")
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

        ax = plot_optimization_history(study)
        ax.figure.set_size_inches(8, 4.5)
        ax.figure.tight_layout()
        ax.figure.savefig(out / "optimization_history.png", dpi=140)
        if len(complete) >= 4:
            ax = plot_param_importances(study)
            ax.figure.set_size_inches(8, 4.5)
            ax.figure.tight_layout()
            ax.figure.savefig(out / "param_importances.png", dpi=140)
    except Exception as exc:  # noqa: BLE001 - plots are optional
        print(f"Plot generation skipped: {exc}")

    print(json.dumps({k: best[k] for k in ("n_complete", "best_value", "untuned_value", "best_params")}, indent=2))
    print(f"Trials written to {models.repo_relative(out / 'trials.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
