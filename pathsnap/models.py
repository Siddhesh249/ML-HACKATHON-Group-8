"""Model zoo, grouped cross-validation and artifact persistence."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from . import config

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
MODEL_FAMILIES = {
    "logistic_regression": "linear / probabilistic",
    "gaussian_nb": "probabilistic",
    "decision_tree": "tree",
    "random_forest": "ensemble",
    "extra_trees": "ensemble",
    "hist_gradient_boosting": "boosting",
    "mlp": "neural network",
    "linear_svm": "kernel / margin",
}


def build_estimator(name: str, params: dict | None = None, seed: int = config.SEED):
    params = dict(params or {})
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=params.get("max_iter", 2000),
                        C=params.get("C", 1.0),
                        class_weight=params.get("class_weight", "balanced"),
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "gaussian_nb":
        return GaussianNB(var_smoothing=params.get("var_smoothing", 1e-9))
    if name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth", 12),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            class_weight=params.get("class_weight", "balanced"),
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight", "balanced_subsample"),
            n_jobs=params.get("n_jobs", -1),
            random_state=seed,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight", "balanced"),
            n_jobs=params.get("n_jobs", -1),
            random_state=seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=params.get("learning_rate", 0.08),
            max_iter=params.get("max_iter", 300),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            min_samples_leaf=params.get("min_samples_leaf", 40),
            l2_regularization=params.get("l2_regularization", 1.0),
            early_stopping=params.get("early_stopping", True),
            validation_fraction=params.get("validation_fraction", 0.1),
            random_state=seed,
        )
    if name == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=params.get("hidden_layer_sizes", (64, 32)),
                        alpha=params.get("alpha", 1e-4),
                        learning_rate_init=params.get("learning_rate_init", 1e-3),
                        max_iter=params.get("max_iter", 200),
                        early_stopping=True,
                        n_iter_no_change=8,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "linear_svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    CalibratedClassifierCV(
                        LinearSVC(
                            C=params.get("C", 1.0),
                            class_weight=params.get("class_weight", "balanced"),
                            max_iter=params.get("max_iter", 5000),
                            random_state=seed,
                        ),
                        cv=params.get("calibration_cv", 3),
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def subsample_negatives(
    y: np.ndarray,
    ratio: float = 3.0,
    seed: int = config.SEED,
) -> np.ndarray:
    """Indices keeping all labelled positives plus a sample of negatives.

    ``y == -1`` rows are always dropped from fitting.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    keep_neg = min(negatives.size, int(max(len(positives), 1) * ratio) + 50)
    if keep_neg < negatives.size:
        negatives = rng.choice(negatives, size=keep_neg, replace=False)
    return np.concatenate([positives, negatives])


def cross_validate_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    params: dict | None = None,
    n_splits: int = 5,
    negative_ratio: float = 3.0,
    seed: int = config.SEED,
    fold_scorer=None,
) -> tuple[np.ndarray, list[dict], float]:
    """Out-of-fold probabilities with GroupKFold over trips.

    Returns ``(oof_prob, fold_reports, train_seconds)``. Rows with ``y == -1``
    are predicted (needed for observation Top-1) but never used for fitting or
    candidate-level metrics.

    ``fold_scorer(fold, estimator, train_idx, test_idx, test_prob)`` may return
    a dict that is merged into that fold's report (per-fold validation and
    training-side metrics for over/underfitting evidence). Its time is not
    counted in ``train_seconds``.
    """
    import time

    y = np.asarray(y)
    groups = np.asarray(groups)
    n = y.size
    oof = np.full(n, np.nan, dtype=float)
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, unique_groups.size)
    if n_splits < 2:
        raise ValueError("Need at least two groups for grouped cross-validation")

    splitter = GroupKFold(n_splits=n_splits)
    fold_reports: list[dict] = []
    train_seconds = 0.0

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        start = time.perf_counter()
        fit_idx = train_idx[y[train_idx] >= 0]
        keep = subsample_negatives(y[fit_idx], ratio=negative_ratio, seed=seed + fold)
        fit_idx = fit_idx[keep]
        estimator = build_estimator(model_name, params, seed=seed + fold)
        estimator.fit(X[fit_idx], y[fit_idx].astype(int))
        fold_seconds = time.perf_counter() - start
        train_seconds += fold_seconds
        prob = estimator.predict_proba(X[test_idx])[:, 1]
        oof[test_idx] = prob
        report = {
            "fold": fold,
            "n_train": int(fit_idx.size),
            "n_test": int(test_idx.size),
            "train_seconds": float(fold_seconds),
            "test_groups": np.unique(groups[test_idx]).tolist(),
        }
        if fold_scorer is not None:
            report.update(fold_scorer(fold, estimator, train_idx, test_idx, prob) or {})
        fold_reports.append(report)

    return oof, fold_reports, train_seconds


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def repo_relative(path: str | Path) -> str:
    """Path relative to the repo root when possible.

    Committed JSON must not leak a machine-specific absolute path (it contains
    the local user name and breaks reproducibility on another laptop).
    """
    path = Path(path).resolve()
    try:
        return str(path.relative_to(config.ROOT))
    except ValueError:
        return path.name


def model_version(model_name: str, params: dict) -> str:
    payload = json.dumps({"name": model_name, "params": params}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def save_artifact(
    path: str | Path,
    model,
    feature_names: tuple[str, ...] | list[str],
    model_name: str,
    params: dict,
    metrics: dict,
    extra_config: dict | None = None,
) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    version = model_version(model_name, params)
    payload = {
        "model": model,
        "feature_names": list(feature_names),
        "feature_version": config.FEATURE_VERSION,
        "model_name": model_name,
        "model_family": MODEL_FAMILIES.get(model_name, "unknown"),
        "params": params,
        "metrics": metrics,
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": extra_config or {},
    }
    joblib.dump(payload, path)
    return payload


def load_artifact(path: str | Path) -> dict:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Not a PathSnap artifact: {path}")
    return payload
