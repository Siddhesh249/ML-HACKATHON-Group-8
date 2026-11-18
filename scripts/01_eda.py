#!/usr/bin/env python3
"""Stage 1 — dataset inventory, quality statistics and EDA figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import config, io, labels, network as net  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(config.DATA_ROOT))
    parser.add_argument("--out-dir", default=str(config.RESULTS_DIR / "eda"))
    parser.add_argument("--max-trips", type=int, default=None)
    parser.add_argument("--no-labels", action="store_true", help="Skip route-alignment diagnostics")
    parser.add_argument("--no-topology", action="store_true", help="Skip connectivity statistics")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root)
    metas = io.discover_trips(root)
    if args.max_trips:
        metas = metas[: args.max_trips]
    if not metas:
        print(f"No trips found under {root}. Place the dataset in data/ or set PATHSNAP_DATA_ROOT.")
        return 1

    rows = []
    all_gaps = []
    for index, meta in enumerate(metas, start=1):
        trip = io.load_trip(meta.directory, with_route=True)
        dt = np.diff(trip.track.t)
        gaps = dt[dt > 0]
        all_gaps.append(gaps)
        row = meta.to_dict()
        _, _, arc_lengths, _, _ = net.arc_geometry(trip.network)
        route_km = float(arc_lengths[trip.route].sum() / 1000) if trip.route is not None else float("nan")
        row.update(
            {
                "track_km": float(
                    np.hypot(
                        np.diff(trip.track.lonlat[:, 0]) * 111320 * np.cos(np.deg2rad(trip.track.lonlat[:, 1].mean())),
                        np.diff(trip.track.lonlat[:, 1]) * 110540,
                    ).sum()
                    / 1000
                ),
                "median_dt_s": float(np.median(gaps)) if gaps.size else float("nan"),
                "max_dt_s": float(gaps.max()) if gaps.size else float("nan"),
                "frac_dt_gt_5s": float((gaps > 5).mean()) if gaps.size else 0.0,
                "route_km": route_km,
            }
        )
        if not args.no_topology:
            row.update(net.topology_stats(trip.network))
        if not args.no_labels and trip.route is not None:
            alignment = labels.align_track_to_route(trip)
            row.update(
                {
                    "label_coverage": alignment.coverage,
                    "label_median_m": float(np.median(alignment.distance_m)),
                    "label_p90_m": float(np.percentile(alignment.distance_m, 90)),
                    "backward_fraction": alignment.backward_fraction,
                    "route_length_m": alignment.route_length_m,
                }
            )
        rows.append(row)
        if index % 10 == 0 or index == len(metas):
            print(f"[{index}/{len(metas)}] {meta.partition}/{meta.trip_id}", flush=True)

    frame = pd.DataFrame(rows)
    # A coarse route length from the aligned polyline is added below when labels run.
    frame.to_csv(out / "inventory.csv", index=False)

    gaps = np.concatenate(all_gaps) if all_gaps else np.array([])

    fig, ax = plt.subplots(figsize=(8, 4))
    frame.boxplot(column="n_points", by="partition", ax=ax)
    ax.set_title("GPS observations per trip")
    ax.set_ylabel("observations")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(out / "points_per_trip.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    if gaps.size:
        ax.hist(np.clip(gaps, 0, 120), bins=60, color="#1565c0")
        ax.set_yscale("log")
    ax.set_title("Sampling intervals (clipped at 120 s)")
    ax.set_xlabel("seconds")
    ax.set_ylabel("count (log)")
    fig.tight_layout()
    fig.savefig(out / "sampling_intervals.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(frame["n_nodes"], frame["n_arcs"], s=18, alpha=0.7, c="#2e7d32")
    ax.set_xlabel("nodes")
    ax.set_ylabel("arcs")
    ax.set_title("Road-network size per trip")
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out / "network_sizes.png", dpi=140)
    plt.close(fig)

    if "label_coverage" in frame:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(frame["label_coverage"].dropna(), bins=30, color="#6a1b9a")
        ax.set_xlabel("fraction of confident observations")
        ax.set_ylabel("trips")
        ax.set_title("Reference-route alignment coverage")
        fig.tight_layout()
        fig.savefig(out / "label_coverage.png", dpi=140)
        plt.close(fig)

    # One representative GPS + network plot (smallest trip for speed).
    representative = frame.sort_values("n_arcs").iloc[0]
    trip = io.load_trip(root / representative["partition"] / representative["trip_id"], with_route=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    arcs = trip.network.arcs[:: max(1, trip.meta.n_arcs // 4000)]
    for arc in arcs:
        a, b = trip.network.arcs[arc]
        ax.plot(
            [trip.network.nodes[a, 0], trip.network.nodes[b, 0]],
            [trip.network.nodes[a, 1], trip.network.nodes[b, 1]],
            color="0.8",
            linewidth=0.4,
        )
    ax.plot(trip.track.lonlat[:, 0], trip.track.lonlat[:, 1], color="#1565c0", linewidth=0.8, label="GPS")
    if trip.route is not None:
        for arc in trip.route:
            a, b = trip.network.arcs[arc]
            ax.plot(
                [trip.network.nodes[a, 0], trip.network.nodes[b, 0]],
                [trip.network.nodes[a, 1], trip.network.nodes[b, 1]],
                color="#2e7d32",
                linewidth=1.0,
                alpha=0.8,
            )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Trip {trip.meta.trip_id}: GPS (blue) + reference route (green)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "representative_trip.png", dpi=150)
    plt.close(fig)

    summary = {
        "data_root": str(root),
        "n_trips": int(len(frame)),
        "partitions": frame["partition"].value_counts().to_dict(),
        "total_points": int(frame["n_points"].sum()),
        "total_nodes": int(frame["n_nodes"].sum()),
        "total_arcs": int(frame["n_arcs"].sum()),
        "median_points": float(frame["n_points"].median()),
        "median_max_gap_s": float(frame["max_dt_s"].median()),
        "trips_with_gap_gt_5s": int((frame["frac_dt_gt_5s"] > 0.001).sum()),
        "median_label_coverage": float(frame["label_coverage"].median()) if "label_coverage" in frame else None,
        "min_label_coverage": float(frame["label_coverage"].min()) if "label_coverage" in frame else None,
    }
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Outputs written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
