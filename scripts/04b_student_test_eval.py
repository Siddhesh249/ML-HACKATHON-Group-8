#!/usr/bin/env python3
"""Stage 4b — student-test evaluation (T22) and tuned vs untuned (T24).

Every model is refit once on all 80 train trips (same negative subsampling as
CV and as the exported artifact) and scored on the 10 student-test trips:
observation Top-1, macro/positive F1, log loss, fit time and inference time.

Tuned vs untuned compares the selected model with default parameters against
the Optuna best parameters on (a) the same GroupKFold CV folds — read from the
Optuna study, where trial 0 is the untuned default — and (b) student-test.

The choice of configuration is made on CV only; student-test is reported, never
used to choose. Evaluator trips are never read.
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

MODELS = [
    "hist_gradient_boosting", "random_forest", "extra_trees", "decision_tree",
    "mlp", "linear_svm", "logistic_regression", "gaussian_nb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", default=str(config.CACHE_DIR / "dataset.npz"))
    parser.add_argument("--test-dataset", default=str(config.CACHE_DIR / "student_test.npz"))
    parser.add_argument("--best-params", default=str(config.RESULTS_DIR / "optuna" / "best_params.json"))
    parser.add_argument("--out-dir", default=str(config.RESULTS_DIR / "models"))
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--data-root", default=str(config.DATA_ROOT), help="for road networks (continuity)")
    return parser.parse_args()


def fit_full(train: dict, name: str, params: dict, ratio: float, seed: int):
    y = train["y"]
    fit_idx = np.flatnonzero(y >= 0)
    keep = models.subsample_negatives(y[fit_idx], ratio=ratio, seed=seed)
    fit_idx = fit_idx[keep]
    est = models.build_estimator(name, params, seed=seed)
    t0 = time.perf_counter()
    est.fit(train["X"][fit_idx], y[fit_idx].astype(int))
    return est, time.perf_counter() - t0, int(fit_idx.size)


def score(test: dict, est) -> tuple[dict, np.ndarray]:
    t0 = time.perf_counter()
    prob = est.predict_proba(test["X"])[:, 1]
    infer = time.perf_counter() - t0
    top1, n_eval, n_correct = dataset.top1_by_mask(
        prob, test["row_obs"], test["row_arc"], test["labels"], test["confidence"]
    )
    cand = metrics.candidate_metrics(test["y"], prob)
    out = {
        "top1_accuracy": float(top1), "top1_n": int(n_eval), "top1_correct": int(n_correct),
        "macro_f1": cand.get("macro_f1"), "f1": cand.get("f1"), "precision": cand.get("precision"),
        "recall": cand.get("recall"), "log_loss": cand.get("log_loss"),
        "inference_seconds_all_trips": float(infer),
        "inference_ms_per_candidate": float(infer * 1000.0 / max(prob.size, 1)),
    }
    return out, prob


def load_networks(test: dict, data_root: str) -> list:
    """Road networks of the student-test trips (no reference route loaded)."""
    from pathsnap import io

    nets = []
    for trip_id in test["trip_ids"]:
        trip_dir = Path(data_root) / "student_test" / str(trip_id)
        nets.append(io.load_trip(trip_dir, with_route=False).network)
    return nets


def continuity(test: dict, prob: np.ndarray, networks: list) -> float:
    """Mean per-trip strict continuity of the classifier-only argmax sequence."""
    arcs = dataset.predicted_sequence_from_probs(prob, test["obs_offsets"], test["row_arc"])
    vals = []
    for t, network in enumerate(networks):
        c = metrics.route_continuity(arcs[test["obs_trip"] == t], network)["strict_continuity"]
        if np.isfinite(c):
            vals.append(c)
    return float(np.mean(vals)) if vals else float("nan")


def per_trip_top1(test: dict, prob: np.ndarray) -> dict:
    res = {}
    for t, trip_id in enumerate(test["trip_ids"]):
        top1, _, _ = dataset.top1_by_mask(prob, test["row_obs"], test["row_arc"], test["labels"],
                                          test["confidence"], test["obs_trip"] == t)
        res[str(trip_id)] = float(top1)
    return res


def main() -> int:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = dataset.load_dataset(args.train_dataset)
    test = dataset.load_dataset(args.test_dataset)
    if not np.array_equal(train["feature_names"], test["feature_names"]):
        raise SystemExit("Train and student-test feature contracts differ; rebuild both datasets.")
    overlap = set(map(str, train["trip_ids"])) & set(map(str, test["trip_ids"]))
    if overlap:
        raise SystemExit(f"Trip leakage between train and student-test: {sorted(overlap)}")

    run_meta = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "git_commit": models.git_commit(),
        "seed": args.seed,
        "partition": "student_test (fit on all 80 train trips)",
        "negative_ratio": args.negative_ratio,
    }

    networks = load_networks(test, args.data_root)

    # ---- T22: every model, default parameters, on student-test -------------
    rows, per_trip = [], {}
    for name in args.models:
        est, fit_s, n_fit = fit_full(train, name, {}, args.negative_ratio, args.seed)
        res, prob = score(test, est)
        res["strict_continuity_classifier_only"] = continuity(test, prob, networks)
        per_trip[name] = per_trip_top1(test, prob)
        rows.append({**run_meta, "model": name, "family": models.MODEL_FAMILIES.get(name),
                     "config": "default (untuned) parameters", "n_fit_rows": n_fit,
                     "fit_seconds": float(fit_s), **res})
        print(f"{name:<24} student-test top1={res['top1_accuracy']:.4f} macroF1={res['macro_f1']:.4f} "
              f"logloss={res['log_loss']:.4f} fit={fit_s:.1f}s", flush=True)

    # Rule-based reference: nearest candidate by distance.
    dist_col = list(config.FEATURE_NAMES).index("log_distance")
    base, _, _ = dataset.top1_by_mask(-test["X"][:, dist_col].astype(float), test["row_obs"], test["row_arc"],
                                      test["labels"], test["confidence"])
    rows.append({**run_meta, "model": "nearest_road_baseline", "family": "rule-based (reference only)",
                 "config": "argmin perpendicular distance", "top1_accuracy": float(base)})
    st = pd.DataFrame(rows).sort_values("top1_accuracy", ascending=False)
    st.to_csv(out / "student_test_comparison.csv", index=False)

    cv = pd.read_csv(out / "model_comparison.csv")
    joined = cv[["model", "family", "top1_accuracy", "macro_f1", "log_loss", "inference_ms_per_candidate"]].rename(
        columns={"top1_accuracy": "cv_top1", "macro_f1": "cv_macro_f1", "log_loss": "cv_log_loss",
                 "inference_ms_per_candidate": "inference_ms_per_candidate_cv"}
    ).merge(st[["model", "top1_accuracy", "macro_f1", "log_loss", "strict_continuity_classifier_only"]].rename(
        columns={"top1_accuracy": "student_top1", "macro_f1": "student_macro_f1", "log_loss": "student_log_loss",
                 "strict_continuity_classifier_only": "student_continuity_classifier_only"}
    ), on="model", how="outer")
    joined["cv_rank"] = joined["cv_top1"].rank(ascending=False, method="min")
    joined["student_rank"] = joined["student_top1"].rank(ascending=False, method="min")
    joined.sort_values("cv_top1", ascending=False).to_csv(out / "cv_vs_student_test.csv", index=False)

    # ---- T24: tuned vs untuned for the selected model ------------------------
    with open(args.best_params, encoding="utf-8") as fh:
        study = json.load(fh)
    name = study["model"]
    configs = {"untuned": {}, "tuned": dict(study["best_params"])}
    tv_rows = []
    for label, params in configs.items():
        cv_metrics = study["untuned_metrics"] if label == "untuned" else study["best_metrics"]
        cv_top1 = study["untuned_value"] if label == "untuned" else study["best_value"]
        tv_rows.append({**run_meta, "model": name, "config": label, "partition": "train CV (GroupKFold %d)" % study["folds"],
                        "params": json.dumps(params) if params else "defaults", "top1_accuracy": cv_top1,
                        "top1_fold_std": cv_metrics.get("top1_fold_std"), "macro_f1": cv_metrics.get("macro_f1"),
                        "f1": cv_metrics.get("f1"), "log_loss": cv_metrics.get("log_loss"),
                        "fit_seconds": cv_metrics.get("fit_seconds"), "source": study["study"]})
        est, fit_s, _ = fit_full(train, name, params, args.negative_ratio, args.seed)
        res, prob = score(test, est)
        res["strict_continuity_classifier_only"] = continuity(test, prob, networks)
        per_trip[f"{name}_{label}"] = per_trip_top1(test, prob)
        tv_rows.append({**run_meta, "model": name, "config": label, "partition": "student_test",
                        "params": json.dumps(params) if params else "defaults", **res,
                        "fit_seconds": float(fit_s), "source": "fit on all train trips"})
    tv = pd.DataFrame(tv_rows)
    for part in tv["partition"].unique():
        m = tv["partition"] == part
        base_row = tv[m & (tv["config"] == "untuned")].iloc[0]
        for col in ("top1_accuracy", "macro_f1", "log_loss"):
            tv.loc[m, f"delta_{col}_vs_untuned"] = tv.loc[m, col] - base_row[col]
    tv.to_csv(out / "tuned_vs_untuned.csv", index=False)

    cv_rows = tv[tv["partition"].str.startswith("train")]
    cv_untuned = cv_rows[cv_rows["config"] == "untuned"].iloc[0]
    cv_tuned = cv_rows[cv_rows["config"] == "tuned"].iloc[0]
    # docs/EXPERIMENTS.md §3: select on CV Top-1 with positive F1 and log loss as
    # guardrails -> tuned must improve Top-1 without worsening either guardrail.
    improves = (cv_tuned["top1_accuracy"] > cv_untuned["top1_accuracy"]) and (
        cv_tuned["log_loss"] <= cv_untuned["log_loss"]) and (cv_tuned["f1"] >= cv_untuned["f1"])
    decision = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(["python3", "scripts/04b_student_test_eval.py", *sys.argv[1:]]),
        "model": name,
        "rule": "EXPERIMENTS.md §3: export tuned only if CV Top-1 improves and neither guardrail (positive F1, log loss) gets worse; student-test is reported, never used to choose",
        "cv_top1_untuned": float(cv_untuned["top1_accuracy"]),
        "cv_top1_tuned": float(cv_tuned["top1_accuracy"]),
        "cv_delta": float(cv_tuned["top1_accuracy"] - cv_untuned["top1_accuracy"]),
        "cv_fold_std_untuned": cv_untuned.get("top1_fold_std"),
        "decision": "export tuned" if improves else "keep untuned",
    }
    with open(out / "tuned_vs_untuned_decision.json", "w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2, default=float)
    pd.DataFrame(per_trip).rename_axis("trip_id").reset_index().to_csv(out / "student_test_per_trip.csv", index=False)

    print("\n" + tv[["config", "partition", "top1_accuracy", "macro_f1", "log_loss", "fit_seconds"]].to_string(index=False))
    print(json.dumps(decision, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
