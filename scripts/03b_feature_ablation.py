#!/usr/bin/env python3
"""Stage 3b — feature-group ablation (T13) and grouped permutation importance.

Ablation: the same model and the same GroupKFold folds, trained on growing
feature sets (distance only -> local -> + relative -> + temporal/graph) and on
leave-one-group-out sets. Only columns change; candidates and labels are
identical, so the Top-1 differences are attributable to the features.

Importance: permutation importance of each feature measured as the drop in
held-out observation Top-1 (the selection metric), averaged over folds and
repeats. Permutation is done within the held-out fold only.
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

from pathsnap import config, dataset, features, metrics, models  # noqa: E402

G = features.FEATURE_GROUPS
ALL = tuple(config.FEATURE_NAMES)


def ablation_sets() -> list[tuple[str, str, tuple[str, ...]]]:
    local, rel, tg = G["local"], G["relative"], G["temporal_graph"]
    return [
        ("distance_only", "cumulative", ("log_distance",)),
        ("local", "cumulative", local),
        ("local+relative", "cumulative", local + rel),
        ("local+relative+temporal_graph (full)", "cumulative", ALL),
        ("full - relative", "leave_one_out", local + tg),
        ("full - temporal_graph", "leave_one_out", local + rel),
        ("full - heading", "leave_one_out", tuple(f for f in ALL if f not in features.HEADING_FEATURES)),
        ("full - graph", "leave_one_out", tuple(f for f in ALL if f not in features.GRAPH_FEATURES)),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(config.CACHE_DIR / "dataset.npz"))
    parser.add_argument("--out-dir", default=str(config.RESULTS_DIR / "models"))
    parser.add_argument("--model", default="hist_gradient_boosting")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--repeats", type=int, default=3, help="Permutation repeats per feature per fold")
    parser.add_argument("--skip-importance", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Redraw figures from the existing CSVs")
    return parser.parse_args()


def fold_top1(data: dict, rows: np.ndarray, prob: np.ndarray) -> float:
    full = np.full(data["y"].size, np.nan)
    full[rows] = prob
    trips = np.unique(data["row_trip"][rows])
    top1, _, _ = dataset.top1_by_mask(
        full, data["row_obs"], data["row_arc"], data["labels"], data["confidence"],
        np.isin(data["obs_trip"], trips),
    )
    return top1


def run_ablation(data: dict, args, run_meta: dict) -> pd.DataFrame:
    rows = []
    for name, kind, feats in ablation_sets():
        cols = features.feature_indices(feats)
        per_fold: list[float] = []

        def scorer(fold, est, train_idx, test_idx, prob):
            t = fold_top1(data, test_idx, prob)
            per_fold.append(t)
            return {"val_top1": t}

        t0 = time.perf_counter()
        oof, _, train_seconds = models.cross_validate_oof(
            data["X"][:, cols], data["y"], data["row_trip"], args.model,
            n_splits=args.folds, negative_ratio=args.negative_ratio, seed=args.seed,
            fold_scorer=scorer,
        )
        top1, n_eval, _ = dataset.top1_from_probs(
            oof, data["obs_offsets"], data["row_arc"], data["labels"], data["confidence"]
        )
        cand = metrics.candidate_metrics(data["y"], oof)
        arr = np.asarray(per_fold)
        row = {
            **run_meta,
            "feature_set": name,
            "kind": kind,
            "n_features": len(cols),
            "features": " ".join(config.FEATURE_NAMES[i] for i in cols),
            "top1_accuracy": float(top1),
            "top1_fold_mean": float(arr.mean()),
            "top1_fold_std": float(arr.std(ddof=1)),
            "macro_f1": cand.get("macro_f1"),
            "f1": cand.get("f1"),
            "log_loss": cand.get("log_loss"),
            "top1_n": int(n_eval),
            "train_seconds": float(train_seconds),
            "wall_seconds": float(time.perf_counter() - t0),
        }
        rows.append(row)
        print(f"{name:<40} n={len(cols):>2} top1={top1:.4f} ±{row['top1_fold_std']:.4f} "
              f"macroF1={row['macro_f1']:.4f} logloss={row['log_loss']:.4f}", flush=True)
    frame = pd.DataFrame(rows)
    full = frame.loc[frame["feature_set"].str.contains("full")].iloc[0]["top1_accuracy"]
    frame["delta_top1_vs_full"] = frame["top1_accuracy"] - full
    return frame


def run_importance(data: dict, args, run_meta: dict) -> pd.DataFrame:
    X, y = data["X"], data["y"]
    rng = np.random.default_rng(args.seed)
    drops: dict[str, list[float]] = {f: [] for f in ALL}
    base_scores: list[float] = []

    def scorer(fold, est, train_idx, test_idx, prob):
        base = fold_top1(data, test_idx, prob)
        base_scores.append(base)
        Xt = X[test_idx].copy()
        for j, feat in enumerate(ALL):
            original = Xt[:, j].copy()
            for _ in range(args.repeats):
                Xt[:, j] = rng.permutation(original)
                drops[feat].append(base - fold_top1(data, test_idx, est.predict_proba(Xt)[:, 1]))
            Xt[:, j] = original
        print(f"  importance fold {fold} done (base top1 {base:.4f})", flush=True)
        return {}

    models.cross_validate_oof(
        X, y, data["row_trip"], args.model, n_splits=args.folds,
        negative_ratio=args.negative_ratio, seed=args.seed, fold_scorer=scorer,
    )
    group_of = {f: g for g, fs in G.items() for f in fs}
    rows = [
        {
            **run_meta,
            "feature": f,
            "group": group_of[f],
            "top1_drop_mean": float(np.mean(v)),
            "top1_drop_std": float(np.std(v, ddof=1)),
            "n_evaluations": len(v),
            "base_top1_fold_mean": float(np.mean(base_scores)),
        }
        for f, v in drops.items()
    ]
    return pd.DataFrame(rows).sort_values("top1_drop_mean", ascending=False)


def plot(ablation: pd.DataFrame, importance: pd.DataFrame | None, out: Path, model: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#3b6ea8" if k == "cumulative" else "#c07a2c" for k in ablation["kind"]]
    ax.barh(ablation["feature_set"], ablation["top1_accuracy"], xerr=ablation["top1_fold_std"],
            color=colors, capsize=3)
    ax.invert_yaxis()
    lo = max(0.0, ablation["top1_accuracy"].min() - 0.05)
    ax.set_xlim(lo, min(1.0, ablation["top1_accuracy"].max() + 0.03))
    ax.set_xlabel("CV observation Top-1 (pooled OOF; error bar = fold std)")
    ax.set_title(f"Feature-group ablation — {model}\nGroupKFold(5) on 80 train trips", fontsize=10)
    for i, v in enumerate(ablation["top1_accuracy"]):
        ax.text(lo + 0.004, i, f"{v:.3f}", va="center", fontsize=8, color="white", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "feature_ablation.png", dpi=150)
    plt.close(fig)

    if importance is None:
        return
    palette = {"local": "#3b6ea8", "relative": "#5a9e6f", "temporal_graph": "#c07a2c"}
    imp = importance.sort_values("top1_drop_mean")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(imp["feature"], imp["top1_drop_mean"], xerr=imp["top1_drop_std"],
            color=[palette[g] for g in imp["group"]], capsize=2)
    ax.set_xlabel("Drop in held-out observation Top-1 when the feature is permuted")
    ax.set_title(f"Grouped permutation importance — {model}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    ax.legend(handles, palette.keys(), loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        imp_path = out / "feature_importance.csv"
        plot(pd.read_csv(out / "feature_ablation.csv"),
             pd.read_csv(imp_path) if imp_path.exists() else None, out, args.model)
        print(f"Redrew figures in {out}")
        return 0
    data = dataset.load_dataset(args.dataset)
    run_meta = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "git_commit": models.git_commit(),
        "seed": args.seed,
        "partition": "train (GroupKFold by trip)",
        "model": args.model,
        "config": "default params (full profile), negative ratio %.1f" % args.negative_ratio,
    }
    ablation = run_ablation(data, args, run_meta)
    ablation.to_csv(out / "feature_ablation.csv", index=False)
    importance = None
    if not args.skip_importance:
        importance = run_importance(data, args, run_meta)
        importance.to_csv(out / "feature_importance.csv", index=False)
    plot(ablation, importance, out, args.model)
    with open(out / "feature_ablation_meta.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                **run_meta,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "command": " ".join(["python3", "scripts/03b_feature_ablation.py", *sys.argv[1:]]),
                "groups": {k: list(v) for k, v in G.items()},
                "repeats": args.repeats,
                "folds": args.folds,
            },
            fh,
            indent=2,
        )
    print(f"Wrote {models.repo_relative(out / 'feature_ablation.csv')}"
          + ("" if importance is None else f" and {models.repo_relative(out / 'feature_importance.csv')}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
