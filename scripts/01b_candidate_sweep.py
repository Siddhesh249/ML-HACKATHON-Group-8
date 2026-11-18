#!/usr/bin/env python3
"""Stage 1b — candidate generation validation: radius / top-K sweep (T11).

For every trip, labels come from the reference-route alignment (D-009). For
each search radius the candidate generator is run once with a large top-K and
**no fallback radii** (to isolate the radius), then every top-K is evaluated
by truncating the distance-sorted shortlist. The production setting (radius
60 m + fallbacks 100/150 m) is evaluated as an extra row.

Miss rate = confident labelled observations whose reference arc is not among
the candidates. It is the ceiling on observation Top-1 (1 - miss).

The configuration is chosen on **train** trips only; student-test rows are a
development check. Evaluator trips are never read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import candidates as cand_mod, config, io, labels  # noqa: E402

RADII = (30.0, 50.0, 60.0, 80.0, 100.0)
TOP_KS = (5, 10, 15, 20, 30)
MAX_K = max(TOP_KS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(config.DATA_ROOT))
    parser.add_argument("--out-dir", default=str(config.RESULTS_DIR / "eda"))
    parser.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    parser.add_argument("--max-trips", type=int, default=None)
    return parser.parse_args()


def truth_rank(cands, truth: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank (0-based) of the truth arc per evaluated observation; -1 if absent.

    Returns ``(rank, n_candidates)`` over the evaluated observations only.
    """
    evaluated = np.flatnonzero(conf & (truth >= 0))
    ranks = np.full(evaluated.size, -1, dtype=np.int64)
    counts = np.diff(cands.offsets)[evaluated]
    for j, i in enumerate(evaluated):
        lo, hi = int(cands.offsets[i]), int(cands.offsets[i + 1])
        hit = np.flatnonzero(cands.arc_ids[lo:hi] == truth[i])
        if hit.size:
            ranks[j] = int(hit[0])
    return ranks, counts


def summarise(ranks: np.ndarray, counts: np.ndarray, k: int) -> dict:
    n = ranks.size
    hit = (ranks >= 0) & (ranks < k)
    kept = np.minimum(counts, k)
    return {
        "n_eval": int(n),
        "n_miss": int(n - hit.sum()),
        "miss_rate": float(1 - hit.mean()) if n else float("nan"),
        "frac_no_candidates": float((counts == 0).mean()) if n else float("nan"),
        "mean_candidates": float(kept.mean()) if n else float("nan"),
        "median_candidates": float(np.median(kept)) if n else float("nan"),
        "p95_candidates": float(np.percentile(kept, 95)) if n else float("nan"),
        "truth_rank0_rate": float((ranks == 0).mean()) if n else float("nan"),
    }


def main() -> int:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metas = []
    for part in args.partitions:
        metas.extend(io.discover_trips(args.data_root, partition=part))
    if args.max_trips:
        metas = metas[: args.max_trips]
    if not metas:
        print("No trips found.")
        return 1

    per_trip = []
    pooled: dict[tuple, list] = {}
    seconds: dict[str, float] = {}
    for index, meta in enumerate(metas, start=1):
        trip = io.load_trip(meta.directory, with_route=True)
        alignment = labels.align_track_to_route(trip)
        truth, conf = alignment.arc_ids, alignment.confidence
        gen = cand_mod.CandidateGenerator(trip.network, radius_m=RADII[0], top_k=MAX_K, radius_fallbacks_m=())
        settings = [(f"r{int(r)}", r, ()) for r in RADII]
        settings.append(("r60+fallback", config.CANDIDATE_RADIUS_M, config.CANDIDATE_RADIUS_FALLBACKS_M))
        for name, radius, fallbacks in settings:
            gen.radius_m, gen.radius_fallbacks_m, gen.top_k = float(radius), tuple(fallbacks), MAX_K
            t0 = time.perf_counter()
            cands = gen.generate(trip.track)
            seconds[name] = seconds.get(name, 0.0) + time.perf_counter() - t0
            ranks, counts = truth_rank(cands, truth, conf)
            pooled.setdefault((meta.partition, name), []).append((ranks, counts))
            for k in TOP_KS:
                per_trip.append({"partition": meta.partition, "trip_id": meta.trip_id, "setting": name,
                                 "radius_m": radius, "fallback": bool(fallbacks), "top_k": k,
                                 **summarise(ranks, counts, k)})
        if index % 10 == 0 or index == len(metas):
            print(f"[{index}/{len(metas)}] {meta.partition}/{meta.trip_id}", flush=True)

    per_trip_frame = pd.DataFrame(per_trip)
    per_trip_frame.to_csv(out / "candidate_sweep_per_trip.csv", index=False)

    rows = []
    for (part, name), chunks in pooled.items():
        ranks = np.concatenate([c[0] for c in chunks])
        counts = np.concatenate([c[1] for c in chunks])
        sub = per_trip_frame[(per_trip_frame["partition"] == part) & (per_trip_frame["setting"] == name)]
        for k in TOP_KS:
            s = summarise(ranks, counts, k)
            trip_miss = sub.loc[sub["top_k"] == k, "miss_rate"]
            rows.append({
                "partition": part,
                "setting": name,
                "radius_m": float(name.split("+")[0][1:]),
                "fallback": "+fallback" in name,
                "top_k": k,
                **s,
                "trip_miss_median": float(trip_miss.median()),
                "trip_miss_max": float(trip_miss.max()),
                "trips_miss_gt_1pct": int((trip_miss > 0.01).sum()),
                "generate_seconds_all_trips": float(seconds[name]),
            })
    sweep = pd.DataFrame(rows).sort_values(["partition", "fallback", "radius_m", "top_k"])
    sweep.to_csv(out / "candidate_sweep.csv", index=False)

    # Choice (train only): smallest shortlist whose miss rate is within 0.05
    # percentage points of the most generous setting without fallback.
    train = sweep[(sweep["partition"] == "train") & (~sweep["fallback"])]
    best = train["miss_rate"].min()
    ok = train[train["miss_rate"] <= best + 0.0005].sort_values(["mean_candidates", "radius_m"])
    pick = ok.iloc[0].to_dict()
    prod = sweep[(sweep["partition"] == "train") & (sweep["fallback"])
                 & (sweep["top_k"] == config.CANDIDATE_TOP_K)].iloc[0].to_dict()
    choice = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(["python3", "scripts/01b_candidate_sweep.py", *sys.argv[1:]]),
        "selection_partition": "train",
        "rule": "smallest mean shortlist with miss rate within 0.05 pp of the best no-fallback setting",
        "best_no_fallback_miss_rate": float(best),
        "rule_pick": {k: pick[k] for k in ("radius_m", "top_k", "miss_rate", "mean_candidates")},
        "production_config": {
            "radius_m": config.CANDIDATE_RADIUS_M,
            "fallbacks_m": list(config.CANDIDATE_RADIUS_FALLBACKS_M),
            "top_k": config.CANDIDATE_TOP_K,
            "miss_rate": prod["miss_rate"],
            "mean_candidates": prod["mean_candidates"],
            "frac_no_candidates": prod["frac_no_candidates"],
        },
        "decision": (
            "Keep the production config (60 m + 100/150 m fallbacks, top-K 15). Top-K, not radius, drives the "
            "miss rate on labelled points; radius >= 30 m looks sufficient only because confident labels are by "
            "definition <= 30 m from the route, so the sweep cannot penalise a small radius. At inference the "
            "pipeline also sees unlabelled points far from the route (GPS-to-route p90 ~48 m in results/eda), "
            "which a 30 m radius without fallback would leave candidate-free. K=15 is the knee of the "
            "miss-rate vs shortlist-size trade-off."
        ),
    }
    with open(out / "candidate_sweep_choice.json", "w", encoding="utf-8") as fh:
        json.dump(choice, fh, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for k in TOP_KS:
        sub = train[train["top_k"] == k].sort_values("radius_m")
        axes[0].plot(sub["radius_m"], 100 * sub["miss_rate"], marker="o", label=f"top-K {k}")
        axes[1].plot(sub["radius_m"], sub["mean_candidates"], marker="o", label=f"top-K {k}")
    axes[0].scatter([config.CANDIDATE_RADIUS_M], [100 * prod["miss_rate"]], marker="*", s=160, color="#c62828",
                    zorder=5, label=f"production (60 m + fallback, K={config.CANDIDATE_TOP_K})")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("search radius (m), no fallback")
    axes[0].set_ylabel("candidate miss rate (%, log)")
    axes[0].set_title("Miss rate vs radius (80 train trips)")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("search radius (m)")
    axes[1].set_ylabel("mean candidates per observation")
    axes[1].set_title("Shortlist size vs radius")
    fig.tight_layout()
    fig.savefig(out / "candidate_sweep.png", dpi=140)
    plt.close(fig)

    print(train[["radius_m", "top_k", "miss_rate", "mean_candidates", "frac_no_candidates"]].round(5).to_string(index=False))
    print(json.dumps(choice, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
