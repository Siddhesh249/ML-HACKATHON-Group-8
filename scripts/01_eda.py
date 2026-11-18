#!/usr/bin/env python3
"""Stage 1 — dataset inventory, quality statistics and EDA figures (T10).

Covers the six recommended exploration outputs of the dataset brief:

1. inventory (trip/file/observation counts, file sizes)      -> inventory.csv, summary.json
2. trajectory length, sampling intervals, missing/invalid      -> inventory.csv, quality_checks.csv, figures
3. GPS coordinate distributions and trajectory patterns        -> gps_locations.png, trajectory_gallery.png
4. interactive Folium maps                                     -> visualisation/make_maps.py (results/maps/)
5. road-network size and connectivity                          -> network_stats.csv, network_*.png
6. data-quality record (issues, assumptions, decisions)        -> flags.csv + docs/DATA_NOTES.md

Raw files are re-read with a tolerant parser for the invalid-value counts, so
a malformed row is counted rather than silently aborting the loader.
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

from pathsnap import config, geo, io, labels, network as net  # noqa: E402

SUFFIXES = (".track", ".nodes", ".arcs", ".route")
EXPECTED_COLS = {".track": 3, ".nodes": 2, ".arcs": 2, ".route": 1}
IMPLAUSIBLE_SPEED_MPS = config.HMM_MAX_SPEED_MPS  # ~200 km/h
LONG_ARC_M = 500.0

# Rule-based anomaly flags (documented in docs/DATA_NOTES.md).
FLAG_RULES = {
    "large_gap": ("max_dt_s", lambda v: v > config.HMM_GAP_RESET_S, f"max sampling gap > {config.HMM_GAP_RESET_S:.0f} s"),
    "implausible_speed": ("n_speed_gt_max", lambda v: v > 0, f"raw step speed > {IMPLAUSIBLE_SPEED_MPS:.0f} m/s"),
    "duplicate_or_backward_time": ("n_dt_le_0", lambda v: v > 0, "repeated or decreasing timestamp"),
    "invalid_gps_rows": ("n_invalid_track_rows", lambda v: v > 0, "non-finite or out-of-range GPS rows"),
    "low_label_coverage": ("label_coverage", lambda v: v < 0.7, "< 70% of points within 30 m of the reference route"),
    "loop_or_out_and_back_route": ("route_order_corr", lambda v: v < 0.8,
                                   "Spearman(time, route progress) < 0.8: route revisits places / not monotone"),
    "route_breaks": ("route_breaks", lambda v: v > 0, "consecutive route arcs do not share a node"),
    "fragmented_network": ("largest_component_fraction", lambda v: v < 0.9, "largest weak component < 90% of nodes"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(config.DATA_ROOT))
    parser.add_argument("--out-dir", default=str(config.RESULTS_DIR / "eda"))
    parser.add_argument("--max-trips", type=int, default=None)
    parser.add_argument("--no-labels", action="store_true", help="Skip route-alignment diagnostics")
    parser.add_argument("--no-topology", action="store_true", help="Skip connectivity statistics")
    return parser.parse_args()


def raw_file_checks(directory: Path, trip_id: str) -> dict:
    """File presence/size/shape and tolerant invalid-value counts."""
    out: dict = {}
    total = 0
    for suffix in SUFFIXES:
        path = directory / f"{trip_id}{suffix}"
        key = suffix[1:]
        out[f"has_{key}"] = path.exists()
        size = path.stat().st_size if path.exists() else 0
        out[f"{key}_bytes"] = int(size)
        total += size
        if not path.exists():
            continue
        arr = np.genfromtxt(path, dtype=float, ndmin=2, invalid_raise=False)
        out[f"{key}_rows"] = int(arr.shape[0])
        out[f"{key}_cols_ok"] = bool(arr.shape[1] == EXPECTED_COLS[suffix]) if arr.size else False
        if suffix == ".track" and arr.size:
            finite = np.isfinite(arr).all(axis=1)
            in_range = (np.abs(arr[:, 0]) <= 180) & (np.abs(arr[:, 1]) <= 90)
            out["n_invalid_track_rows"] = int((~(finite & in_range)).sum())
            out["n_zero_coord_rows"] = int(((arr[:, 0] == 0) & (arr[:, 1] == 0)).sum())
        if suffix in (".arcs", ".route") and arr.size:
            out[f"{key}_integer_valued"] = bool(np.all(np.isfinite(arr)) and np.all(arr == np.round(arr)))
    out["n_files"] = int(sum(out[f"has_{s[1:]}"] for s in SUFFIXES))
    out["total_bytes"] = int(total)
    return out


def track_stats(trip) -> tuple[dict, np.ndarray, np.ndarray]:
    t = trip.track.t
    dt = np.diff(t)
    xy, _ = geo.project_lonlat(trip.track.lonlat)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    pos = dt > 0
    speed = np.where(pos, step / np.where(pos, dt, 1.0), np.nan)
    gaps = dt[pos]
    ext = xy.max(axis=0) - xy.min(axis=0)
    stats = {
        "duration_min": float((t[-1] - t[0]) / 60.0),
        "track_km": float(step.sum() / 1000.0),
        "extent_x_km": float(ext[0] / 1000.0),
        "extent_y_km": float(ext[1] / 1000.0),
        "center_lon": float(trip.track.lonlat[:, 0].mean()),
        "center_lat": float(trip.track.lonlat[:, 1].mean()),
        "timestamps_sorted": bool(np.all(dt >= 0)),
        "n_dt_le_0": int((dt <= 0).sum()),
        "frac_dt_eq_1": float(np.mean(dt == 1)) if dt.size else float("nan"),
        "median_dt_s": float(np.median(gaps)) if gaps.size else float("nan"),
        "max_dt_s": float(gaps.max()) if gaps.size else 0.0,
        "n_gaps_gt_5s": int((gaps > 5).sum()),
        "n_gaps_gt_30s": int((gaps > 30).sum()),
        "frac_dt_gt_5s": float((gaps > 5).mean()) if gaps.size else 0.0,
        "n_stationary_steps": int((step < 0.5).sum()),
        "median_speed_mps": float(np.nanmedian(speed)) if speed.size else float("nan"),
        "p99_speed_mps": float(np.nanpercentile(speed, 99)) if np.isfinite(speed).any() else float("nan"),
        "n_speed_gt_max": int(np.nansum(speed > IMPLAUSIBLE_SPEED_MPS)),
    }
    return stats, gaps, speed[np.isfinite(speed)]


def network_stats(trip) -> tuple[dict, np.ndarray]:
    network = trip.network
    _, _, lengths, _, _ = net.arc_geometry(network)
    topo = net.topology_stats(network)
    n_comp, comp = net.components(network)
    sizes = np.bincount(comp)
    out_deg, in_deg = net.node_degrees(network)
    codes = network.arcs[:, 0].astype(np.int64) * network.n_nodes + network.arcs[:, 1]
    stats = {
        **topo,
        "largest_component_fraction": float(sizes.max() / network.n_nodes) if sizes.size else 0.0,
        "nodes_without_arcs": int(((out_deg + in_deg) == 0).sum()),
        "self_loop_arcs": int((network.arcs[:, 0] == network.arcs[:, 1]).sum()),
        "duplicate_arcs": int(codes.size - np.unique(codes).size),
        "zero_length_arcs": int((lengths < 0.01).sum()),
        "reciprocal_fraction": float(net.reciprocal_mask(network).mean()),
        "arc_len_median_m": float(np.median(lengths)),
        "arc_len_p95_m": float(np.percentile(lengths, 95)),
        "arc_len_max_m": float(lengths.max()),
        "frac_arcs_gt_500m": float((lengths > LONG_ARC_M).mean()),
        "network_km": float(lengths.sum() / 1000.0),
    }
    return stats, lengths


def route_stats(trip, lengths: np.ndarray) -> dict:
    route = trip.route
    if route is None or route.size == 0:
        return {"route_km": float("nan")}
    arcs = trip.network.arcs[route]
    connected = arcs[:-1, 1] == arcs[1:, 0]
    breaks = np.flatnonzero(~connected)
    nodes_xy, _ = geo.project_lonlat(trip.network.nodes)
    gap = np.linalg.norm(nodes_xy[arcs[breaks, 1]] - nodes_xy[arcs[breaks + 1, 0]], axis=1) if breaks.size else np.zeros(0)
    return {
        "route_km": float(lengths[route].sum() / 1000.0),
        "route_connected_fraction": float(connected.mean()) if connected.size else 1.0,
        "route_breaks": int(breaks.size),
        "route_break_max_m": float(gap.max()) if gap.size else 0.0,
        "route_repeated_arcs": int(route.size - np.unique(route).size),
    }


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_network(ax, trip, max_arcs: int = 6000) -> None:
    arc_ids = np.arange(trip.network.n_arcs)[:: max(1, trip.network.n_arcs // max_arcs)]
    nodes = trip.network.nodes
    for arc in arc_ids:
        a, b = trip.network.arcs[arc]
        ax.plot([nodes[a, 0], nodes[b, 0]], [nodes[a, 1], nodes[b, 1]], color="0.82", linewidth=0.4)


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

    # Directories that exist but were skipped by discovery (missing files / parse errors).
    found = {(m.partition, m.trip_id) for m in metas}
    skipped = [
        f"{p.parent.name}/{p.name}"
        for p in sorted(root.glob("*/*"))
        if p.is_dir() and (p.parent.name, p.name) not in found and not args.max_trips
    ]

    rows, all_gaps, all_speed, all_arc_len, all_route_dist = [], [], [], [], []
    trips_cache = {}
    for index, meta in enumerate(metas, start=1):
        trip = io.load_trip(meta.directory, with_route=True)
        row = {k: v for k, v in meta.to_dict().items() if k != "bbox"}
        row.update(raw_file_checks(meta.directory, meta.trip_id))
        tstats, gaps, speed = track_stats(trip)
        row.update(tstats)
        all_gaps.append(gaps)
        all_speed.append(speed)
        nstats, lengths = network_stats(trip)
        if not args.no_topology:
            row.update(nstats)
        all_arc_len.append(lengths)
        row.update(route_stats(trip, lengths))
        if not args.no_labels and trip.route is not None:
            alignment = labels.align_track_to_route(trip)
            order = labels.route_order_diagnostics(labels.project_onto_route(trip)["progress_m"])
            row.update(
                {
                    "route_order_corr": order["order_corr"],
                    "route_backward_jumps_gt_100m": order["backward_jumps"],
                }
            )
            row.update(
                {
                    "label_coverage": alignment.coverage,
                    "label_median_m": float(np.median(alignment.distance_m)),
                    "label_p90_m": float(np.percentile(alignment.distance_m, 90)),
                    "backward_fraction": alignment.backward_fraction,
                    "route_length_m": alignment.route_length_m,
                }
            )
            all_route_dist.append(alignment.distance_m)
        rows.append(row)
        trips_cache[(meta.partition, meta.trip_id)] = trip
        if index % 10 == 0 or index == len(metas):
            print(f"[{index}/{len(metas)}] {meta.partition}/{meta.trip_id}", flush=True)

    frame = pd.DataFrame(rows)

    # ---- flags (output 6) --------------------------------------------------
    flag_rows = []
    for _, r in frame.iterrows():
        for flag, (col, rule, why) in FLAG_RULES.items():
            if col in r and pd.notna(r[col]) and rule(r[col]):
                flag_rows.append({"partition": r["partition"], "trip_id": r["trip_id"], "flag": flag,
                                  "value": r[col], "rule": why})
    flags = pd.DataFrame(flag_rows, columns=["partition", "trip_id", "flag", "value", "rule"])
    flags.to_csv(out / "flags.csv", index=False)
    frame["flags"] = frame["trip_id"].map(flags.groupby("trip_id")["flag"].apply(lambda s: ";".join(s))).fillna("")
    frame.to_csv(out / "inventory.csv", index=False)

    net_cols = ["partition", "trip_id", "n_nodes", "n_arcs", "n_components", "largest_component",
                "largest_component_fraction", "isolated_nodes", "nodes_without_arcs", "self_loop_arcs",
                "duplicate_arcs", "zero_length_arcs", "reciprocal_fraction", "max_out_degree",
                "max_in_degree", "arc_len_median_m", "arc_len_p95_m", "arc_len_max_m",
                "frac_arcs_gt_500m", "network_km"]
    frame[[c for c in net_cols if c in frame]].to_csv(out / "network_stats.csv", index=False)

    quality = {
        "trips_discovered": int(len(frame)),
        "trip_dirs_skipped": skipped,
        "trips_with_all_4_files": int((frame["n_files"] == 4).sum()),
        "trips_missing_route": frame.loc[~frame["has_route"], "trip_id"].tolist(),
        "column_count_ok_all_files": bool(frame[[c for c in frame if c.endswith("_cols_ok")]].all().all()),
        "invalid_track_rows_total": int(frame["n_invalid_track_rows"].sum()),
        "zero_coord_rows_total": int(frame["n_zero_coord_rows"].sum()),
        "trips_unsorted_timestamps": frame.loc[~frame["timestamps_sorted"], "trip_id"].tolist(),
        "non_positive_dt_total": int(frame["n_dt_le_0"].sum()),
        "implausible_speed_steps_total": int(frame["n_speed_gt_max"].sum()),
        "self_loop_arcs_total": int(frame.get("self_loop_arcs", pd.Series([0])).sum()),
        "duplicate_arcs_total": int(frame.get("duplicate_arcs", pd.Series([0])).sum()),
        "zero_length_arcs_total": int(frame.get("zero_length_arcs", pd.Series([0])).sum()),
    }
    pd.DataFrame([{"check": k, "value": json.dumps(v) if isinstance(v, list) else v}
                  for k, v in quality.items()]).to_csv(out / "quality_checks.csv", index=False)

    gaps = np.concatenate(all_gaps) if all_gaps else np.array([])
    speed = np.concatenate(all_speed) if all_speed else np.array([])
    arc_len = np.concatenate(all_arc_len) if all_arc_len else np.array([])
    route_dist = np.concatenate(all_route_dist) if all_route_dist else np.array([])
    colors = {"train": "#3b6ea8", "student_test": "#c07a2c"}

    # ---- figures -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    frame.boxplot(column="n_points", by="partition", ax=ax)
    ax.set_title("GPS observations per trip")
    ax.set_ylabel("observations")
    fig.suptitle("")
    save(fig, out / "points_per_trip.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if gaps.size:
        axes[0].hist(np.clip(gaps, 0, 120), bins=60, color="#3b6ea8")
        axes[0].set_yscale("log")
    axes[0].set_title("Sampling intervals (clipped at 120 s)")
    axes[0].set_xlabel("seconds between fixes")
    axes[0].set_ylabel("count (log)")
    for part, g in frame.groupby("partition"):
        axes[1].scatter(g["duration_min"], g["track_km"], s=18, alpha=0.75, label=part, color=colors.get(part))
    axes[1].set_xlabel("trip duration (min)")
    axes[1].set_ylabel("GPS path length (km)")
    axes[1].set_title("Trajectory length vs duration")
    axes[1].legend()
    save(fig, out / "sampling_intervals.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for part, g in frame.groupby("partition"):
        axes[0].scatter(g["center_lon"], g["center_lat"], s=22, alpha=0.75, label=part, color=colors.get(part))
    axes[0].set_xlabel("longitude (°)")
    axes[0].set_ylabel("latitude (°)")
    axes[0].set_title("Trip centroids (each trip has its own local network)")
    axes[0].legend()
    axes[1].scatter(frame["extent_x_km"], frame["extent_y_km"], s=18, alpha=0.75, color="#5a9e6f")
    axes[1].set_xlabel("east–west extent (km)")
    axes[1].set_ylabel("north–south extent (km)")
    axes[1].set_title("Geographical extent per trip")
    save(fig, out / "gps_locations.png")

    # Trajectory gallery: 12 trips spread over the size range, local metres.
    order = frame.sort_values("track_km").reset_index(drop=True)
    pick = order.iloc[np.linspace(0, len(order) - 1, min(12, len(order))).astype(int)]
    fig, axes = plt.subplots(3, 4, figsize=(13, 9.5))
    for ax, (_, r) in zip(axes.ravel(), pick.iterrows()):
        trip = trips_cache[(r["partition"], r["trip_id"])]
        xy, _ = geo.project_lonlat(trip.track.lonlat)
        ax.plot(xy[:, 0] / 1000, xy[:, 1] / 1000, linewidth=0.8, color=colors.get(r["partition"]))
        ax.plot(xy[0, 0] / 1000, xy[0, 1] / 1000, "o", color="#2e7d32", markersize=4)
        ax.plot(xy[-1, 0] / 1000, xy[-1, 1] / 1000, "s", color="#c62828", markersize=4)
        ax.set_title(f"{r['partition']}/{r['trip_id']} · {r['track_km']:.1f} km", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle("Trajectory patterns (local metres, km; ● start, ■ end)")
    save(fig, out / "trajectory_gallery.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(frame["n_nodes"], frame["n_arcs"], s=18, alpha=0.7, color="#2e7d32")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("nodes")
    axes[0].set_ylabel("arcs")
    axes[0].set_title("Road-network size per trip")
    if "n_components" in frame:
        axes[1].scatter(frame["n_components"], frame["largest_component_fraction"], s=18, alpha=0.7, color="#6a1b9a")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("weakly connected components")
        axes[1].set_ylabel("largest component / all nodes")
        axes[1].set_title("Network connectivity per trip")
    save(fig, out / "network_sizes.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    if arc_len.size:
        ax.hist(np.clip(arc_len, 0.01, None), bins=np.logspace(-2, np.log10(max(arc_len.max(), 1.0)), 60), color="#2e7d32")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(LONG_ARC_M, color="#c62828", linestyle="--", linewidth=1, label=f"{LONG_ARC_M:.0f} m")
        ax.legend()
    ax.set_xlabel("arc length (m, log)")
    ax.set_ylabel("arcs (log)")
    ax.set_title(f"Arc-length distribution (all trips, median {np.median(arc_len):.1f} m)")
    save(fig, out / "arc_lengths.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if speed.size:
        axes[0].hist(np.clip(speed, 0, 80), bins=80, color="#3b6ea8")
        axes[0].set_yscale("log")
        axes[0].axvline(IMPLAUSIBLE_SPEED_MPS, color="#c62828", linestyle="--", linewidth=1)
    axes[0].set_xlabel("raw step speed (m/s, clipped at 80)")
    axes[0].set_ylabel("steps (log)")
    axes[0].set_title("Raw GPS step speed")
    if route_dist.size:
        axes[1].hist(np.clip(route_dist, 0, 100), bins=100, color="#6a1b9a")
        axes[1].set_yscale("log")
        axes[1].axvline(config.LABEL_MAX_DISTANCE_M, color="#c62828", linestyle="--", linewidth=1,
                        label=f"label threshold {config.LABEL_MAX_DISTANCE_M:.0f} m")
        axes[1].legend()
    axes[1].set_xlabel("distance from GPS point to nearest reference-route arc (m, clipped at 100)")
    axes[1].set_ylabel("observations (log)")
    axes[1].set_title("GPS-to-route distance")
    save(fig, out / "speed_and_route_distance.png")

    if "label_coverage" in frame:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(frame["label_coverage"].dropna(), bins=30, color="#6a1b9a")
        ax.set_xlabel("fraction of observations within 30 m of the reference route")
        ax.set_ylabel("trips")
        ax.set_title("Reference-route alignment coverage")
        save(fig, out / "label_coverage.png")

    # Representative trip: smallest network, GPS + reference route.
    rep = frame.sort_values("n_arcs").iloc[0]
    trip = trips_cache[(rep["partition"], rep["trip_id"])]
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_network(ax, trip)
    ax.plot(trip.track.lonlat[:, 0], trip.track.lonlat[:, 1], color="#1565c0", linewidth=0.9, label="GPS")
    if trip.route is not None:
        nodes = trip.network.nodes
        for i, arc in enumerate(trip.route):
            a, b = trip.network.arcs[arc]
            ax.plot([nodes[a, 0], nodes[b, 0]], [nodes[a, 1], nodes[b, 1]], color="#2e7d32",
                    linewidth=1.6, alpha=0.8, label="reference route" if i == 0 else None)
    ax.set_aspect(1 / np.cos(np.deg2rad(trip.track.lonlat[:, 1].mean())), adjustable="datalim")
    ax.set_title(f"Trip {rep['partition']}/{trip.meta.trip_id}: GPS, network (grey), reference route")
    ax.set_xlabel("longitude (°)")
    ax.set_ylabel("latitude (°)")
    ax.legend(loc="best")
    save(fig, out / "representative_trip.png")

    by_part = frame.groupby("partition")
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(["python3", "scripts/01_eda.py", *sys.argv[1:]]),
        "n_trips": int(len(frame)),
        "partitions": frame["partition"].value_counts().to_dict(),
        "observations_by_partition": {k: int(v) for k, v in by_part["n_points"].sum().items()},
        "total_points": int(frame["n_points"].sum()),
        "total_bytes": int(frame["total_bytes"].sum()),
        "total_nodes": int(frame["n_nodes"].sum()),
        "total_arcs": int(frame["n_arcs"].sum()),
        "points_per_trip": {"min": int(frame["n_points"].min()), "median": float(frame["n_points"].median()),
                            "max": int(frame["n_points"].max())},
        "nodes_per_trip": {"min": int(frame["n_nodes"].min()), "median": float(frame["n_nodes"].median()),
                           "max": int(frame["n_nodes"].max())},
        "arcs_per_trip": {"min": int(frame["n_arcs"].min()), "median": float(frame["n_arcs"].median()),
                          "max": int(frame["n_arcs"].max())},
        "route_arcs_per_trip": {"min": int(frame["n_route_arcs"].min()),
                                "median": float(frame["n_route_arcs"].median()),
                                "max": int(frame["n_route_arcs"].max())},
        "track_km": {"min": float(frame["track_km"].min()), "median": float(frame["track_km"].median()),
                     "max": float(frame["track_km"].max())},
        "duration_min": {"median": float(frame["duration_min"].median()), "max": float(frame["duration_min"].max())},
        "sampling": {
            "frac_steps_dt_eq_1s": float(np.mean(gaps == 1)) if gaps.size else None,
            "median_dt_s": float(np.median(gaps)) if gaps.size else None,
            "p99_dt_s": float(np.percentile(gaps, 99)) if gaps.size else None,
            "max_dt_s": float(gaps.max()) if gaps.size else None,
            "trips_with_gap_gt_5s": int((frame["n_gaps_gt_5s"] > 0).sum()),
            "trips_with_gap_gt_30s": int((frame["n_gaps_gt_30s"] > 0).sum()),
        },
        "speed": {"median_mps": float(np.median(speed)) if speed.size else None,
                  "p99_mps": float(np.percentile(speed, 99)) if speed.size else None,
                  "steps_gt_max": int((speed > IMPLAUSIBLE_SPEED_MPS).sum())},
        "arc_length_m": {"median": float(np.median(arc_len)), "p95": float(np.percentile(arc_len, 95)),
                         "max": float(arc_len.max()), "frac_gt_500m": float(np.mean(arc_len > LONG_ARC_M))},
        "connectivity": {
            "median_components": float(frame["n_components"].median()) if "n_components" in frame else None,
            "max_components": int(frame["n_components"].max()) if "n_components" in frame else None,
            "min_largest_component_fraction": float(frame["largest_component_fraction"].min())
            if "largest_component_fraction" in frame else None,
            "median_reciprocal_fraction": float(frame["reciprocal_fraction"].median())
            if "reciprocal_fraction" in frame else None,
        },
        "route_connected_all": bool((frame["route_connected_fraction"] == 1.0).all())
        if "route_connected_fraction" in frame else None,
        "trips_with_route_breaks": int((frame["route_breaks"] > 0).sum()) if "route_breaks" in frame else None,
        "route_break_max_m": float(frame["route_break_max_m"].max()) if "route_break_max_m" in frame else None,
        "route_order_corr_median": float(frame["route_order_corr"].median()) if "route_order_corr" in frame else None,
        "gps_to_route_m": {"median": float(np.median(route_dist)), "p90": float(np.percentile(route_dist, 90)),
                           "frac_within_30m": float(np.mean(route_dist <= config.LABEL_MAX_DISTANCE_M))}
        if route_dist.size else None,
        "median_label_coverage": float(frame["label_coverage"].median()) if "label_coverage" in frame else None,
        "min_label_coverage": float(frame["label_coverage"].min()) if "label_coverage" in frame else None,
        "quality": quality,
        "flag_counts": flags["flag"].value_counts().to_dict(),
        "flagged_trips": sorted(flags["trip_id"].unique().tolist()),
    }
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(json.dumps({k: summary[k] for k in ("n_trips", "observations_by_partition", "sampling", "flag_counts")},
                     indent=2, default=str))
    print(f"Outputs written to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
