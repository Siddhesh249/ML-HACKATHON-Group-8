"""Typed contracts shared by every module.

These are intentionally small dataclasses (not pydantic) so the training code
stays lightweight. The API has its own pydantic schemas in backend/schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class PathSnapError(Exception):
    """Base error for the project."""


class DataError(PathSnapError):
    """Raised when a trip cannot be loaded or fails validation."""


class ModelNotReady(PathSnapError):
    """Raised when an inference artifact is missing or incomplete."""


@dataclass(frozen=True)
class TripMeta:
    trip_id: str
    partition: str
    directory: Path
    n_points: int
    n_nodes: int
    n_arcs: int
    n_route_arcs: int
    has_route: bool
    bbox: tuple[float, float, float, float]  # lon_min, lon_max, lat_min, lat_max

    def to_dict(self) -> dict:
        return {
            "trip_id": self.trip_id,
            "partition": self.partition,
            "n_points": self.n_points,
            "n_nodes": self.n_nodes,
            "n_arcs": self.n_arcs,
            "n_route_arcs": self.n_route_arcs,
            "has_route": self.has_route,
            "bbox": list(self.bbox),
        }


@dataclass
class Track:
    """GPS observations in (lon, lat) order plus timestamps in seconds."""

    lonlat: np.ndarray  # (N, 2)
    t: np.ndarray  # (N,)

    def __len__(self) -> int:
        return int(self.lonlat.shape[0])


@dataclass
class Network:
    """Directed road network. Node/arc IDs are the row indices."""

    nodes: np.ndarray  # (M, 2) lon/lat
    arcs: np.ndarray  # (A, 2) int from_node, to_node

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_arcs(self) -> int:
        return int(self.arcs.shape[0])


@dataclass
class TripData:
    meta: TripMeta
    track: Track
    network: Network
    route: np.ndarray | None = None  # reference arc sequence, server-side only


@dataclass
class Alignment:
    """Per-observation labels derived from the reference route."""

    arc_ids: np.ndarray  # (N,) reference arc per observation, -1 if unlabeled
    confidence: np.ndarray  # (N,) bool
    distance_m: np.ndarray  # (N,) distance to the assigned route segment
    progress_m: np.ndarray  # (N,) monotonic arclength along the route
    route_length_m: float
    coverage: float  # fraction of route length covered by confident points
    backward_fraction: float  # fraction of raw projections moving backwards

    @property
    def labeled(self) -> np.ndarray:
        return self.confidence


@dataclass
class CandidateSet:
    """Ragged candidate sets flattened into arrays.

    Candidates for observation ``i`` are
    ``offsets[i]:offsets[i + 1]`` in every parallel array.
    """

    offsets: np.ndarray  # (N + 1,) int
    arc_ids: np.ndarray  # (C,) int
    distance_m: np.ndarray  # (C,) float
    projection_ratio: np.ndarray  # (C,) float in [0, 1]
    projection_xy: np.ndarray  # (C, 2) local metres
    arc_heading_deg: np.ndarray  # (C,) float
    origin: tuple[float, float] = (0.0, 0.0)  # projection origin (lon0, lat0)

    def __len__(self) -> int:
        return int(self.offsets.shape[0] - 1)

    def slice(self, i: int) -> tuple[np.ndarray, ...]:
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        return (
            self.arc_ids[lo:hi],
            self.distance_m[lo:hi],
            self.projection_ratio[lo:hi],
            self.projection_xy[lo:hi],
            self.arc_heading_deg[lo:hi],
        )


@dataclass
class FeatureMatrix:
    X: np.ndarray  # (C, F) float32
    y: np.ndarray  # (C,) int8, -1 when the observation is unlabeled
    obs_index: np.ndarray  # (C,) int32
    arc_ids: np.ndarray  # (C,) int32
    trip_id: str


@dataclass
class Prediction:
    trip_id: str
    arc_ids: np.ndarray  # one predicted arc per observation, -1 if unmatched
    probabilities: np.ndarray  # (N,) probability of the chosen arc
    mode: str  # "classifier" or "hmm"
    timings_ms: dict[str, float] = field(default_factory=dict)
    model_version: str = "unknown"

    def route_arcs(self) -> np.ndarray:
        """Predicted arcs with consecutive duplicates and unmatched points removed."""
        out: list[int] = []
        for arc in self.arc_ids:
            arc = int(arc)
            if arc < 0:
                continue
            if not out or out[-1] != arc:
                out.append(arc)
        return np.asarray(out, dtype=np.int32)


@dataclass
class EvalResult:
    trip_id: str
    metrics: dict[str, float]
    n_observations: int
    n_labeled: int
    n_matched: int
