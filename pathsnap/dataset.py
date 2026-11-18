"""Dataset construction: candidates + labels + features, cached for training.

The cache is git-ignored. Rebuild with ``make dataset``. A per-trip cache makes
the build resumable when one large network is slow.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import candidates as cand_mod
from . import config, features, io, labels, metrics
from .contracts import CandidateSet, FeatureMatrix, TripData


def build_trip(
    trip: TripData,
    use_graph: bool = config.USE_GRAPH_FEATURES,
    radius_m: float = config.CANDIDATE_RADIUS_M,
    top_k: int = config.CANDIDATE_TOP_K,
) -> dict:
    generator = cand_mod.CandidateGenerator(trip.network, radius_m=radius_m, top_k=top_k)
    candidates = generator.generate(trip.track)
    alignment = labels.align_track_to_route(trip)
    fm = features.build_features(
        trip,
        candidates,
        alignment,
        features.FeatureOptions(use_graph=use_graph),
    )
    miss, miss_eval = cand_mod.candidate_miss_rate(
        candidates, alignment.arc_ids, alignment.confidence
    )
    counts = np.diff(candidates.offsets)
    stats = {
        "trip_id": trip.meta.trip_id,
        "partition": trip.meta.partition,
        "n_points": int(len(trip.track)),
        "n_labeled": int(alignment.confidence.sum()),
        "coverage": float(alignment.coverage),
        "backward_fraction": float(alignment.backward_fraction),
        "alignment_median_m": float(np.median(alignment.distance_m)) if alignment.distance_m.size else float("nan"),
        "alignment_p90_m": float(np.percentile(alignment.distance_m, 90)) if alignment.distance_m.size else float("nan"),
        "candidate_miss_rate": float(miss),
        "candidate_miss_n": int(miss_eval),
        "median_candidates": float(np.median(counts)) if counts.size else 0.0,
        "mean_candidates": float(np.mean(counts)) if counts.size else 0.0,
        "n_rows": int(fm.X.shape[0]),
        "n_positive": int((fm.y == 1).sum()),
        "n_negative": int((fm.y == 0).sum()),
        "n_ignored": int((fm.y < 0).sum()),
        "n_nodes": trip.meta.n_nodes,
        "n_arcs": trip.meta.n_arcs,
    }
    return {
        "X": fm.X,
        "y": fm.y,
        "obs_index": fm.obs_index,
        "arc_ids": fm.arc_ids,
        "labels": alignment.arc_ids,
        "confidence": alignment.confidence,
        "alignment_distance_m": alignment.distance_m,
        "stats": stats,
    }


def _trip_cache_path(cache_dir: Path, trip_id: str) -> Path:
    return cache_dir / "trips" / f"{trip_id}.npz"


def build_trip_cached(
    trip: TripData,
    cache_dir: Path,
    use_graph: bool = config.USE_GRAPH_FEATURES,
    force: bool = False,
) -> dict:
    path = _trip_cache_path(cache_dir, trip.meta.trip_id)
    if path.exists() and not force:
        loaded = np.load(path, allow_pickle=True)
        stats = json.loads(str(loaded["stats"]))
        return {
            "X": loaded["X"],
            "y": loaded["y"],
            "obs_index": loaded["obs_index"],
            "arc_ids": loaded["arc_ids"],
            "labels": loaded["labels"],
            "confidence": loaded["confidence"],
            "alignment_distance_m": loaded["alignment_distance_m"],
            "stats": stats,
        }
    payload = build_trip(trip, use_graph=use_graph)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=payload["X"],
        y=payload["y"],
        obs_index=payload["obs_index"],
        arc_ids=payload["arc_ids"],
        labels=payload["labels"],
        confidence=payload["confidence"],
        alignment_distance_m=payload["alignment_distance_m"],
        stats=np.asarray(json.dumps(payload["stats"])),
    )
    return payload


def build_dataset(
    data_root: str | Path | None = None,
    partitions: tuple[str, ...] = ("train",),
    out_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    limit: int | None = None,
    use_graph: bool = config.USE_GRAPH_FEATURES,
    force: bool = False,
) -> dict:
    root = Path(data_root) if data_root is not None else config.DATA_ROOT
    cache = Path(cache_dir) if cache_dir is not None else config.CACHE_DIR
    out = Path(out_path) if out_path is not None else cache / "dataset.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    metas = []
    for partition in partitions:
        metas.extend(io.discover_trips(root, partition=partition))
    if limit is not None:
        metas = metas[:limit]
    if not metas:
        raise RuntimeError(f"No trips found under {root} for partitions {partitions}")

    trip_ids: list[str] = []
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    row_obs_parts: list[np.ndarray] = []
    row_arc_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    conf_parts: list[np.ndarray] = []
    stats_rows: list[dict] = []
    obs_offset = 0

    for index, meta in enumerate(metas, start=1):
        trip = io.load_trip(meta.directory, with_route=True)
        payload = build_trip_cached(trip, cache, use_graph=use_graph, force=force)
        fm_X = payload["X"]
        n_obs = int(len(trip.track))
        trip_ids.append(meta.trip_id)
        x_parts.append(fm_X)
        y_parts.append(payload["y"])
        row_obs_parts.append(payload["obs_index"].astype(np.int64) + obs_offset)
        row_arc_parts.append(payload["arc_ids"])
        label_parts.append(payload["labels"])
        conf_parts.append(payload["confidence"])
        stats_rows.append(payload["stats"])
        obs_offset += n_obs
        print(
            f"[{index:>3}/{len(metas)}] {meta.partition}/{meta.trip_id}: "
            f"{n_obs} points, {fm_X.shape[0]} rows, "
            f"coverage={payload['stats']['coverage']:.3f}, "
            f"miss={payload['stats']['candidate_miss_rate']:.3f}",
            flush=True,
        )

    X = np.vstack(x_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.int8)
    row_obs = np.concatenate(row_obs_parts).astype(np.int64)
    row_arc = np.concatenate(row_arc_parts).astype(np.int32)
    labels_all = np.concatenate(label_parts).astype(np.int64)
    confidence_all = np.concatenate(conf_parts).astype(bool)

    total_obs = int(obs_offset)
    counts = np.bincount(row_obs, minlength=total_obs)
    obs_offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    obs_counts = [len(part) for part in label_parts]
    obs_trip = np.repeat(np.arange(len(trip_ids)), np.asarray(obs_counts, dtype=np.int64))
    row_trip = obs_trip[row_obs]
    np.savez_compressed(
        out,
        X=X,
        y=y,
        row_obs=row_obs,
        row_arc=row_arc,
        row_trip=row_trip,
        obs_offsets=obs_offsets,
        labels=labels_all,
        confidence=confidence_all,
        obs_trip=obs_trip,
        trip_ids=np.asarray(trip_ids),
        feature_names=np.asarray(list(config.FEATURE_NAMES)),
        use_graph=np.asarray([bool(use_graph)]),
    )
    meta = {
        "path": str(out),
        "partitions": list(partitions),
        "n_trips": len(trip_ids),
        "n_observations": int(total_obs),
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "n_ignored": int((y < 0).sum()),
        "use_graph": bool(use_graph),
        "trip_ids": trip_ids,
        "stats": stats_rows,
    }
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def load_dataset(path: str | Path) -> dict:
    loaded = np.load(path, allow_pickle=False)
    return {key: loaded[key] for key in loaded.files}


def write_stat_reports(meta: dict, labels_dir: str | Path, eda_dir: str | Path | None = None) -> None:
    """Write per-trip label coverage and candidate miss-rate tables.

    These are evidence files: the report and viva must quote them rather than a
    number from memory.
    """
    import pandas as pd

    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(meta.get("stats", []))
    if frame.empty:
        return
    frame.to_csv(labels_dir / "coverage.csv", index=False)

    summary = {
        "n_trips": int(len(frame)),
        "n_observations": int(frame["n_points"].sum()),
        "n_labeled": int(frame["n_labeled"].sum()),
        "mean_coverage": float(frame["coverage"].mean()),
        "median_coverage": float(frame["coverage"].median()),
        "min_coverage": float(frame["coverage"].min()),
        "mean_candidate_miss_rate": float(frame["candidate_miss_rate"].mean()),
        "median_candidate_miss_rate": float(frame["candidate_miss_rate"].median()),
        "max_candidate_miss_rate": float(frame["candidate_miss_rate"].max()),
        "weighted_candidate_miss_rate": float(
            (frame["candidate_miss_rate"] * frame["candidate_miss_n"]).sum()
            / max(frame["candidate_miss_n"].sum(), 1)
        ),
        "trips_coverage_below_0_7": frame.loc[frame["coverage"] < 0.7, "trip_id"].tolist(),
        "trips_miss_above_0_2": frame.loc[frame["candidate_miss_rate"] > 0.2, "trip_id"].tolist(),
    }
    with open(labels_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if eda_dir is not None:
        eda_dir = Path(eda_dir)
        eda_dir.mkdir(parents=True, exist_ok=True)
        columns = [
            "trip_id",
            "partition",
            "n_points",
            "n_labeled",
            "coverage",
            "candidate_miss_rate",
            "median_candidates",
            "alignment_median_m",
            "alignment_p90_m",
        ]
        frame[[c for c in columns if c in frame.columns]].to_csv(
            eda_dir / "candidate_miss_summary.csv", index=False
        )


def observation_frame(dataset: dict) -> dict:
    """Build observation-level arrays needed by Top-1 and sequence metrics."""
    return {
        "obs_offsets": dataset["obs_offsets"],
        "row_obs": dataset["row_obs"],
        "row_arc": dataset["row_arc"],
        "labels": dataset["labels"],
        "confidence": dataset["confidence"],
        "obs_trip": dataset["obs_trip"],
    }


def top1_from_probs(
    probs: np.ndarray,
    obs_offsets: np.ndarray,
    row_arc: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
) -> tuple[float, int, int]:
    n_eval = 0
    n_correct = 0
    for obs in range(obs_offsets.size - 1):
        if not confidence[obs] or labels[obs] < 0:
            continue
        lo, hi = int(obs_offsets[obs]), int(obs_offsets[obs + 1])
        n_eval += 1
        if hi <= lo:
            continue
        j = lo + int(np.argmax(probs[lo:hi]))
        if int(row_arc[j]) == int(labels[obs]):
            n_correct += 1
    return (n_correct / n_eval if n_eval else float("nan")), n_eval, n_correct


def predicted_sequence_from_probs(
    probs: np.ndarray,
    obs_offsets: np.ndarray,
    row_arc: np.ndarray,
) -> np.ndarray:
    out = np.full(obs_offsets.size - 1, -1, dtype=np.int64)
    for obs in range(obs_offsets.size - 1):
        lo, hi = int(obs_offsets[obs]), int(obs_offsets[obs + 1])
        if hi > lo:
            out[obs] = int(row_arc[lo + int(np.argmax(probs[lo:hi]))])
    return out
