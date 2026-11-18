#!/usr/bin/env python3
"""Stage 6 — refit the selected model and export the inference artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import config, dataset, models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(config.CACHE_DIR / "dataset.npz"))
    parser.add_argument("--selected", default=str(config.RESULTS_DIR / "models" / "selected_model.json"))
    parser.add_argument("--best-params", default=str(config.RESULTS_DIR / "optuna" / "best_params.json"))
    parser.add_argument("--artifact-dir", default=str(config.ARTIFACT_DIR))
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.selected, encoding="utf-8") as fh:
        selected = json.load(fh)
    model_name = selected["model"]
    params = dict(selected.get("params", {}))

    best_path = Path(args.best_params)
    if best_path.exists():
        with open(best_path, encoding="utf-8") as fh:
            tuned = json.load(fh)
        if tuned.get("model") == model_name:
            params = dict(tuned.get("best_params", params))
            selected["tuned"] = {"value": tuned.get("best_value"), "folds": tuned.get("folds")}

    data = dataset.load_dataset(args.dataset)
    X, y = data["X"], data["y"]
    fit_idx = np.flatnonzero(y >= 0)
    keep = models.subsample_negatives(y[fit_idx], ratio=args.negative_ratio, seed=args.seed)
    fit_idx = fit_idx[keep]
    estimator = models.build_estimator(model_name, params, seed=args.seed)
    print(f"Refitting {model_name} on {fit_idx.size} rows (params={params})")
    estimator.fit(X[fit_idx], y[fit_idx].astype(int))

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    extra_config = {
        "use_graph_features": bool(data["use_graph"][0]) if "use_graph" in data else config.USE_GRAPH_FEATURES,
        "candidate_radius_m": config.CANDIDATE_RADIUS_M,
        "candidate_top_k": config.CANDIDATE_TOP_K,
        "smooth_window": config.SMOOTH_WINDOW,
        "label_max_distance_m": config.LABEL_MAX_DISTANCE_M,
    }
    metrics = {k: float(v) for k, v in selected.items() if isinstance(v, (int, float))}
    payload = models.save_artifact(
        artifact_dir / "model.joblib",
        estimator,
        feature_names=config.FEATURE_NAMES,
        model_name=model_name,
        params=params,
        metrics=metrics,
        extra_config=extra_config,
    )

    feature_config = {
        "feature_version": config.FEATURE_VERSION,
        "feature_names": list(config.FEATURE_NAMES),
        "smooth_window": config.SMOOTH_WINDOW,
        "candidate_radius_m": config.CANDIDATE_RADIUS_M,
        "candidate_top_k": config.CANDIDATE_TOP_K,
        "label_max_distance_m": config.LABEL_MAX_DISTANCE_M,
        "use_graph_features": extra_config["use_graph_features"],
    }
    with open(artifact_dir / "feature_config.json", "w", encoding="utf-8") as fh:
        json.dump(feature_config, fh, indent=2)

    hmm_path = artifact_dir / "hmm_config.json"
    if not hmm_path.exists():
        with open(hmm_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "beta_m": config.HMM_BETA_M,
                    "sigma_m": config.HMM_SIGMA_M,
                    "lam": config.HMM_LAMBDA,
                    "max_speed_mps": config.HMM_MAX_SPEED_MPS,
                    "gap_reset_s": config.HMM_GAP_RESET_S,
                    "uturn_penalty": config.HMM_UTURN_PENALTY,
                    "transition_slack_m": 300.0,
                },
                fh,
                indent=2,
            )

    model_card = {k: v for k, v in payload.items() if k != "model"}
    model_card["dataset"] = str(Path(args.dataset).resolve())
    model_card["n_fit_rows"] = int(fit_idx.size)
    model_card["n_positive"] = int((y[fit_idx] == 1).sum())
    model_card["n_negative"] = int((y[fit_idx] == 0).sum())
    with open(artifact_dir / "model_card.json", "w", encoding="utf-8") as fh:
        json.dump(model_card, fh, indent=2, default=str)

    print(f"Artifact written to {artifact_dir / 'model.joblib'} (version {payload['model_version']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
