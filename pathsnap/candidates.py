"""Candidate generation: shortlist nearby road arcs for every GPS observation."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import config, geo, network as net
from .contracts import CandidateSet, Network, Track


class CandidateGenerator:
    """Spatial index over densely sampled arc geometry.

    Sampling the arcs (rather than only their endpoints) is essential: the
    provided network contains contracted chords that cut across curved roads,
    so an arc can be closest to a point while its endpoints are far away.
    """

    def __init__(
        self,
        road_network: Network,
        radius_m: float = config.CANDIDATE_RADIUS_M,
        top_k: int = config.CANDIDATE_TOP_K,
        sample_step_m: float = config.ARC_SAMPLE_STEP_M,
        max_samples_per_arc: int = config.ARC_MAX_SAMPLES,
        radius_fallbacks_m: tuple[float, ...] = config.CANDIDATE_RADIUS_FALLBACKS_M,
        origin: tuple[float, float] | None = None,
    ) -> None:
        self.network = road_network
        self.radius_m = float(radius_m)
        self.top_k = int(top_k)
        self.radius_fallbacks_m = tuple(radius_fallbacks_m)
        self.a_xy, self.b_xy, self.lengths_m, self.heading_deg, self.origin = net.arc_geometry(
            road_network, origin=origin
        )
        self._build_index(sample_step_m, max_samples_per_arc)

    def _build_index(self, step_m: float, max_samples: int) -> None:
        lengths = self.lengths_m
        counts = np.clip(np.ceil(lengths / max(step_m, 1e-3)).astype(np.int64) + 1, 2, max_samples)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        total = int(offsets[-1])

        arc_of_sample = np.repeat(np.arange(lengths.size), counts)
        within = np.arange(total) - offsets[arc_of_sample]
        denom = np.maximum(counts[arc_of_sample] - 1, 1)
        frac = (within / denom)[:, None]

        samples = self.a_xy[arc_of_sample] + frac * (self.b_xy[arc_of_sample] - self.a_xy[arc_of_sample])
        self.sample_xy = samples
        self.sample_arc = arc_of_sample
        self.tree = cKDTree(samples)

    def _query_arc_ids(self, point_xy: np.ndarray, radius: float) -> np.ndarray:
        idx = self.tree.query_ball_point(point_xy, r=radius)
        if not idx:
            return np.empty(0, dtype=np.int64)
        return np.unique(self.sample_arc[np.asarray(idx, dtype=np.int64)])

    def candidates_for_point(self, point_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(arc_ids, distances, ratios)`` for one observation."""
        radii = (self.radius_m, *self.radius_fallbacks_m)
        for radius in radii:
            arc_ids = self._query_arc_ids(point_xy, radius)
            if arc_ids.size:
                break
        if arc_ids.size == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=float),
                np.empty(0, dtype=float),
            )

        dist, ratio, _ = geo.point_segment_distance(
            np.asarray(point_xy, dtype=float)[None, :],
            self.a_xy[arc_ids],
            self.b_xy[arc_ids],
        )
        order = np.argsort(dist[0])[: self.top_k]
        return arc_ids[order], dist[0, order], ratio[0, order]

    def generate(self, track: Track) -> CandidateSet:
        """Generate candidates for every observation in ``track``."""
        xy, _ = geo.project_lonlat(track.lonlat, origin=self.origin)
        n = xy.shape[0]
        offsets = np.zeros(n + 1, dtype=np.int64)
        arc_chunks: list[np.ndarray] = []
        dist_chunks: list[np.ndarray] = []
        ratio_chunks: list[np.ndarray] = []
        proj_chunks: list[np.ndarray] = []
        heading_chunks: list[np.ndarray] = []
        cursor = 0
        for i in range(n):
            arc_ids, dist, ratio = self.candidates_for_point(xy[i])
            k = arc_ids.size
            offsets[i + 1] = cursor + k
            if k:
                arc_chunks.append(arc_ids)
                dist_chunks.append(dist)
                ratio_chunks.append(ratio)
                a = self.a_xy[arc_ids]
                b = self.b_xy[arc_ids]
                proj_chunks.append(a + ratio[:, None] * (b - a))
                heading_chunks.append(self.heading_deg[arc_ids])
            cursor += k

        def _cat(chunks: list[np.ndarray], dtype=float) -> np.ndarray:
            if not chunks:
                return np.empty(0, dtype=dtype)
            return np.concatenate(chunks).astype(dtype, copy=False)

        return CandidateSet(
            offsets=offsets,
            arc_ids=_cat(arc_chunks, np.int64),
            distance_m=_cat(dist_chunks, float),
            projection_ratio=_cat(ratio_chunks, float),
            projection_xy=_cat(proj_chunks, float),
            arc_heading_deg=_cat(heading_chunks, float),
            origin=self.origin,
        )


def candidate_miss_rate(
    candidates: CandidateSet,
    labels: np.ndarray,
    confidence: np.ndarray,
) -> tuple[float, int]:
    """Fraction of confident labelled observations whose truth is not a candidate."""
    n_eval = 0
    n_miss = 0
    for i in range(len(candidates)):
        if not confidence[i] or labels[i] < 0:
            continue
        n_eval += 1
        lo, hi = int(candidates.offsets[i]), int(candidates.offsets[i + 1])
        if labels[i] not in candidates.arc_ids[lo:hi]:
            n_miss += 1
    return (n_miss / n_eval if n_eval else float("nan")), n_eval
