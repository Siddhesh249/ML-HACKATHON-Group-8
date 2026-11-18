"""Feature engineering for (observation, candidate arc) pairs.

The feature contract is defined in :mod:`pathsnap.config`. All features are
relative to the current trip and candidate set: no coordinates, node IDs or arc
IDs are used, because every trip ships a different local network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config, geo, network as net
from .contracts import Alignment, CandidateSet, FeatureMatrix, TripData


# Feature groups used by the ablation (T13). Every name in
# ``config.FEATURE_NAMES`` belongs to exactly one group.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    # Properties of this (observation, arc) pair on its own.
    "local": (
        "log_distance",
        "projection_ratio",
        "abs_heading_diff",
        "cos_heading_diff",
        "log_arc_length",
        "log_dist_to_arc_start",
        "log_dist_to_arc_end",
        "node_degree_from",
        "node_degree_to",
        "has_reciprocal_arc",
    ),
    # Position of the candidate relative to the other candidates.
    "relative": (
        "distance_ratio_best",
        "candidate_rank",
        "candidate_count",
    ),
    # Context from the previous observation: network reachability and motion.
    "temporal_graph": (
        "log_graph_dist_prev",
        "same_arc_as_prev",
        "log_graph_minus_gps_displacement",
        "log_speed",
        "log_gap_seconds",
    ),
}

HEADING_FEATURES: tuple[str, ...] = ("abs_heading_diff", "cos_heading_diff")
GRAPH_FEATURES: tuple[str, ...] = (
    "log_graph_dist_prev",
    "same_arc_as_prev",
    "log_graph_minus_gps_displacement",
)


def feature_indices(names) -> list[int]:
    """Column indices of ``names`` in the frozen feature order."""
    lookup = {name: i for i, name in enumerate(config.FEATURE_NAMES)}
    missing = [n for n in names if n not in lookup]
    if missing:
        raise KeyError(f"Unknown features: {missing}")
    return sorted(lookup[n] for n in names)


@dataclass
class FeatureOptions:
    use_graph: bool = config.USE_GRAPH_FEATURES
    smooth_window: int = config.SMOOTH_WINDOW
    corridor_margin_m: float = 600.0
    graph_limit_m: float = 3000.0


class FeatureBuilder:
    """Precomputes trip-level geometry and builds the candidate feature matrix."""

    def __init__(self, trip: TripData, candidates: CandidateSet, options: FeatureOptions | None = None):
        self.trip = trip
        self.candidates = candidates
        self.options = options or FeatureOptions()
        self.a_xy, self.b_xy, self.lengths, self.heading, _ = net.arc_geometry(
            trip.network, origin=candidates.origin
        )
        self.out_degree, self.in_degree = net.node_degrees(trip.network)
        self.reciprocal = net.reciprocal_mask(trip.network)

        self.smooth_lonlat, self.smooth_heading, self.speed = geo.smooth_track(
            trip.track.lonlat, trip.track.t, window=self.options.smooth_window
        )
        self.track_xy, _ = geo.project_lonlat(self.smooth_lonlat, origin=candidates.origin)

        self._graph = None
        self._local_index = None
        self._dist_cache: dict[int, np.ndarray] = {}
        if self.options.use_graph and self.track_xy.shape[0]:
            corridor = net.corridor_subgraph(
                trip.network,
                self.track_xy,
                origin=candidates.origin,
                margin_m=self.options.corridor_margin_m,
            )
            self._graph = corridor["graph"]
            self._local_index = corridor["local_index"]

    # ------------------------------------------------------------------
    def _graph_distances(self, source_global: int, limit_m: float) -> np.ndarray | None:
        if self._graph is None or self._local_index is None:
            return None
        source_local = int(self._local_index[source_global]) if source_global >= 0 else -1
        if source_local < 0:
            return None
        cached = self._dist_cache.get(source_local)
        if cached is None:
            cached = net.shortest_distances(self._graph, source_local, self.options.graph_limit_m)
            self._dist_cache[source_local] = cached
        return cached

    def build(self, alignment: Alignment | None = None) -> FeatureMatrix:
        cand = self.candidates
        n_obs = len(cand)
        if n_obs == 0:
            return FeatureMatrix(
                X=np.empty((0, len(config.FEATURE_NAMES)), dtype=np.float32),
                y=np.empty(0, dtype=np.int8),
                obs_index=np.empty(0, dtype=np.int32),
                arc_ids=np.empty(0, dtype=np.int32),
                trip_id=self.trip.meta.trip_id,
            )

        # Previous best candidate per observation (by raw distance).
        prev_best = np.full(n_obs, -1, dtype=np.int64)
        for i in range(n_obs):
            lo, hi = int(cand.offsets[i]), int(cand.offsets[i + 1])
            if hi > lo:
                prev_best[i] = int(cand.arc_ids[lo + int(np.argmin(cand.distance_m[lo:hi]))])

        x_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []
        obs_rows: list[np.ndarray] = []
        arc_rows: list[np.ndarray] = []
        sentinel = float(np.log1p(self.options.graph_limit_m + 500.0))

        for i in range(n_obs):
            lo, hi = int(cand.offsets[i]), int(cand.offsets[i + 1])
            k = hi - lo
            if k == 0:
                continue
            arc_ids = cand.arc_ids[lo:hi].astype(np.int64)
            dist = cand.distance_m[lo:hi]
            ratio = np.clip(cand.projection_ratio[lo:hi], 0.0, 1.0)
            arc_len = self.lengths[arc_ids]
            heading = self.heading[arc_ids]

            order = np.argsort(dist, kind="stable")
            rank = np.empty(k, dtype=float)
            rank[order] = np.arange(k, dtype=float)
            best = float(dist[order[0]]) if k else 0.0

            heading_diff = np.abs(geo.angle_difference_deg(self.smooth_heading[i], heading))

            prev_arc = int(prev_best[i - 1]) if i > 0 else -1
            same_prev = (arc_ids == prev_arc).astype(float) if prev_arc >= 0 else np.zeros(k)
            gps_disp = float(np.linalg.norm(self.track_xy[i] - self.track_xy[i - 1])) if i > 0 else 0.0
            gap_s = float(max(self.trip.track.t[i] - self.trip.track.t[i - 1], 0.0)) if i > 0 else 0.0

            if self.options.use_graph and prev_arc >= 0:
                source = int(self.trip.network.arcs[prev_arc, 1])
                limit = max(self.options.graph_limit_m, 4.0 * gps_disp + 300.0)
                dist_row = self._graph_distances(source, limit)
                if dist_row is None:
                    graph_dist = np.full(k, np.inf)
                else:
                    targets = self._local_index[self.trip.network.arcs[arc_ids, 0]]
                    valid = targets >= 0
                    graph_dist = np.full(k, np.inf)
                    graph_dist[valid] = dist_row[targets[valid]]
                    graph_dist[valid] += ratio[valid] * arc_len[valid]
                reachable = np.isfinite(graph_dist)
                log_graph = np.where(reachable, np.log1p(graph_dist), sentinel)
                log_graph_delta = np.where(
                    reachable, np.log1p(np.abs(graph_dist - gps_disp)), sentinel
                )
            else:
                log_graph = np.zeros(k)
                log_graph_delta = np.zeros(k)

            degree_from = self.out_degree[self.trip.network.arcs[arc_ids, 0]].astype(float)
            degree_to = self.in_degree[self.trip.network.arcs[arc_ids, 1]].astype(float)

            feats = np.column_stack(
                [
                    np.log1p(dist),
                    dist / max(best, 1e-6),
                    rank,
                    ratio,
                    heading_diff,
                    np.cos(np.deg2rad(heading_diff)),
                    np.log1p(arc_len),
                    np.log1p(np.maximum(ratio * arc_len, 0.0)),
                    np.log1p(np.maximum((1.0 - ratio) * arc_len, 0.0)),
                    degree_from,
                    degree_to,
                    self.reciprocal[arc_ids].astype(float),
                    log_graph,
                    same_prev,
                    log_graph_delta,
                    np.full(k, np.log1p(max(self.speed[i], 0.0))),
                    np.full(k, np.log1p(gap_s)),
                    np.full(k, float(k)),
                ]
            ).astype(np.float32)

            if alignment is not None:
                if not alignment.confidence[i] or alignment.arc_ids[i] < 0:
                    y = np.full(k, -1, dtype=np.int8)
                else:
                    truth = int(alignment.arc_ids[i])
                    if truth in arc_ids:
                        y = (arc_ids == truth).astype(np.int8)
                    else:
                        # The truth is missing from the candidate set; exclude
                        # the whole observation instead of teaching all-negative.
                        y = np.full(k, -1, dtype=np.int8)
            else:
                y = np.zeros(k, dtype=np.int8)

            x_rows.append(feats)
            y_rows.append(y)
            obs_rows.append(np.full(k, i, dtype=np.int32))
            arc_rows.append(arc_ids.astype(np.int32))

        if not x_rows:
            return FeatureMatrix(
                X=np.empty((0, len(config.FEATURE_NAMES)), dtype=np.float32),
                y=np.empty(0, dtype=np.int8),
                obs_index=np.empty(0, dtype=np.int32),
                arc_ids=np.empty(0, dtype=np.int32),
                trip_id=self.trip.meta.trip_id,
            )

        return FeatureMatrix(
            X=np.vstack(x_rows),
            y=np.concatenate(y_rows),
            obs_index=np.concatenate(obs_rows),
            arc_ids=np.concatenate(arc_rows),
            trip_id=self.trip.meta.trip_id,
        )


def build_features(
    trip: TripData,
    candidates: CandidateSet,
    alignment: Alignment | None = None,
    options: FeatureOptions | None = None,
) -> FeatureMatrix:
    return FeatureBuilder(trip, candidates, options).build(alignment)
