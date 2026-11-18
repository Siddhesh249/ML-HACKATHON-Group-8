#!/usr/bin/env python3
"""Stage 1c — route-label coverage table and visual checks (T12).

Labels follow D-009: each GPS observation is labelled with the nearest
reference-route arc; observations further than ``LABEL_MAX_DISTANCE_M`` from
the route are low-confidence and excluded from fitting. Monotonic progress is
reported only as a diagnostic.

Writes ``results/labels/coverage.csv`` (one row per trip), ``summary.json``
(overall and per partition) and ``visual_check_<trip>.png`` for the trips
passed with ``--visual`` (defaults: clean, out-and-back, two low-coverage and
one route-break trip).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import candidates as cand_mod, config, geo, io, labels  # noqa: E402

UNRELIABLE_COVERAGE = 0.7
DEFAULT_VISUAL = ["train/00000079", "train/00000011", "train/00000033", "train/00000054", "train/00000019"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(config.DATA_ROOT))
    parser.add_argument("--out-dir", default=str(config.RESULTS_DIR / "labels"))
    parser.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    parser.add_argument("--visual", nargs="*", default=DEFAULT_VISUAL, help="partition/trip_id to plot")
    return parser.parse_args()


def trip_row(trip) -> tuple[dict, object, dict]:
    alignment = labels.align_track_to_route(trip)
    proj = labels.project_onto_route(trip)
    order = labels.route_order_diagnostics(proj["progress_m"])
    gen = cand_mod.CandidateGenerator(trip.network)
    cands = gen.generate(trip.track)
    miss, n_eval = cand_mod.candidate_miss_rate(cands, alignment.arc_ids, alignment.confidence)
    arcs = trip.network.arcs[trip.route]
    breaks = int((arcs[:-1, 1] != arcs[1:, 0]).sum())
    d = alignment.distance_m
    row = {
        "trip_id": trip.meta.trip_id,
        "partition": trip.meta.partition,
        "n_points": int(d.size),
        "n_labeled": int(alignment.confidence.sum()),
        "coverage": float(alignment.coverage),
        "alignment_median_m": float(np.median(d)),
        "alignment_p90_m": float(np.percentile(d, 90)),
        "frac_within_10m": float(np.mean(d <= 10)),
        "backward_fraction": float(alignment.backward_fraction),
        "route_order_corr": order["order_corr"],
        "route_backward_jumps_gt_100m": order["backward_jumps"],
        "route_arcs": int(trip.route.size),
        "route_breaks": breaks,
        "route_length_km": alignment.route_length_m / 1000.0,
        "n_distinct_labels": int(np.unique(alignment.arc_ids[alignment.confidence]).size),
        "frac_route_arcs_labelled": float(
            np.isin(trip.route, alignment.arc_ids[alignment.confidence]).mean()
        ),
        "candidate_miss_rate": float(miss),
        "candidate_miss_n": int(n_eval),
        "median_candidates": float(np.median(np.diff(cands.offsets))),
    }
    reasons = []
    if row["coverage"] < UNRELIABLE_COVERAGE:
        reasons.append(f"coverage<{UNRELIABLE_COVERAGE}")
    if row["route_order_corr"] < 0.8:
        reasons.append("loop/out-and-back route (diagnostic only)")
    if breaks:
        reasons.append(f"{breaks} route break(s)")
    row["reliable_reference"] = row["coverage"] >= UNRELIABLE_COVERAGE
    row["notes"] = "; ".join(reasons)
    return row, alignment, proj


def plot_check(trip, alignment, row: dict, path: Path) -> None:
    xy, origin = geo.project_lonlat(trip.track.lonlat)
    nodes_xy, _ = geo.project_lonlat(trip.network.nodes, origin=origin)
    lo, hi = xy.min(axis=0) - 400, xy.max(axis=0) + 400
    arcs = trip.network.arcs
    a, b = nodes_xy[arcs[:, 0]], nodes_xy[arcs[:, 1]]
    inside = np.all((a >= lo) & (a <= hi), axis=1) | np.all((b >= lo) & (b <= hi), axis=1)
    idx = np.flatnonzero(inside)
    if idx.size > 8000:
        idx = idx[np.linspace(0, idx.size - 1, 8000).astype(int)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for i in idx:
        ax.plot([a[i, 0] / 1000, b[i, 0] / 1000], [a[i, 1] / 1000, b[i, 1] / 1000], color="0.85", linewidth=0.4)
    r = arcs[trip.route]
    for j, (u, v) in enumerate(r):
        ax.plot([nodes_xy[u, 0] / 1000, nodes_xy[v, 0] / 1000], [nodes_xy[u, 1] / 1000, nodes_xy[v, 1] / 1000],
                color="#2e7d32", linewidth=2.2, alpha=0.6, label="reference route" if j == 0 else None)
    conf = alignment.confidence
    ax.scatter(xy[conf, 0] / 1000, xy[conf, 1] / 1000, s=2, color="#1565c0", label="GPS, labelled (≤30 m)")
    ax.scatter(xy[~conf, 0] / 1000, xy[~conf, 1] / 1000, s=4, color="#c62828", label="GPS, excluded (>30 m)")
    ax.plot(xy[0, 0] / 1000, xy[0, 1] / 1000, "o", color="black", markersize=6, label="start")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(lo[0] / 1000, hi[0] / 1000)
    ax.set_ylim(lo[1] / 1000, hi[1] / 1000)
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    ax.legend(fontsize=7, loc="best")
    ax.set_title(f"{row['partition']}/{row['trip_id']}: coverage {row['coverage']:.1%}, "
                 f"order corr {row['route_order_corr']:.2f}, breaks {row['route_breaks']}", fontsize=10)

    ax = axes[1]
    t = (trip.track.t - trip.track.t[0]) / 60.0
    ax.plot(t, np.clip(alignment.distance_m, 0, 500), linewidth=0.7, color="#6a1b9a")
    ax.axhline(config.LABEL_MAX_DISTANCE_M, color="#c62828", linestyle="--", linewidth=1, label="30 m threshold")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("minutes since start")
    ax.set_ylabel("distance to nearest route arc (m, clipped 500)")
    ax.set_title("Alignment distance over time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    visual = set(args.visual or [])
    rows = []
    for part in args.partitions:
        for meta in io.discover_trips(args.data_root, partition=part):
            trip = io.load_trip(meta.directory, with_route=True)
            row, alignment, _ = trip_row(trip)
            rows.append(row)
            key = f"{part}/{meta.trip_id}"
            if key in visual:
                plot_check(trip, alignment, row, out / f"visual_check_{meta.trip_id}.png")
                print(f"visual check -> visual_check_{meta.trip_id}.png", flush=True)
        print(f"{part}: {sum(r['partition'] == part for r in rows)} trips", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "coverage.csv", index=False)

    def block(f: pd.DataFrame) -> dict:
        return {
            "n_trips": int(len(f)),
            "n_observations": int(f["n_points"].sum()),
            "n_labeled": int(f["n_labeled"].sum()),
            "pooled_coverage": float(f["n_labeled"].sum() / max(f["n_points"].sum(), 1)),
            "mean_coverage": float(f["coverage"].mean()),
            "median_coverage": float(f["coverage"].median()),
            "min_coverage": float(f["coverage"].min()),
            "median_alignment_m": float(f["alignment_median_m"].median()),
            "mean_candidate_miss_rate": float(f["candidate_miss_rate"].mean()),
            "weighted_candidate_miss_rate": float(
                (f["candidate_miss_rate"] * f["candidate_miss_n"]).sum() / max(f["candidate_miss_n"].sum(), 1)),
            "max_candidate_miss_rate": float(f["candidate_miss_rate"].max()),
            "unreliable_trips": f.loc[~f["reliable_reference"], "trip_id"].tolist(),
            "loop_or_out_and_back_trips": f.loc[f["route_order_corr"] < 0.8, "trip_id"].tolist(),
            "route_break_trips": f.loc[f["route_breaks"] > 0, "trip_id"].tolist(),
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(["python3", "scripts/01c_label_coverage.py", *sys.argv[1:]]),
        "policy": "D-009: nearest reference-route arc; confident if <= %.0f m; monotonicity diagnostic only"
        % config.LABEL_MAX_DISTANCE_M,
        "candidate_config": {"radius_m": config.CANDIDATE_RADIUS_M,
                             "fallbacks_m": list(config.CANDIDATE_RADIUS_FALLBACKS_M),
                             "top_k": config.CANDIDATE_TOP_K},
        "unreliable_rule": f"coverage < {UNRELIABLE_COVERAGE}",
        "visual_checks": sorted(visual),
        "all": block(frame),
        "by_partition": {p: block(g) for p, g in frame.groupby("partition")},
    }
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: summary["by_partition"][k] for k in summary["by_partition"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
