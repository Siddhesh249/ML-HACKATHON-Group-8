"""Server-side scoring. This module is the only place that reads ``.route``.

The API calls :func:`evaluate_predictions` *after* a prediction exists, then
returns only the aggregate metrics to the UI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config, geo, labels, metrics
from .contracts import Alignment, CandidateSet, EvalResult, Network, TripData


def evaluate_predictions(
    trip: TripData,
    candidates: CandidateSet,
    alignment: Alignment,
    predicted_arcs: np.ndarray,
) -> EvalResult:
    point_top1, n_eval, n_correct = metrics.pointwise_accuracy(
        predicted_arcs, alignment.arc_ids, alignment.confidence
    )
    miss, n_miss_eval = metrics.candidate_miss_rate(
        candidates, alignment.arc_ids, alignment.confidence
    )
    node_xy, _ = geo.project_lonlat(trip.network.nodes)
    continuity = metrics.route_continuity(predicted_arcs, trip.network, node_xy=node_xy, slack_m=2.0)
    predicted_route = metrics.dedupe_sequence(predicted_arcs)
    reference_route = metrics.dedupe_sequence(trip.route if trip.route is not None else np.empty(0))
    sequence = metrics.sequence_metrics(predicted_route, reference_route)

    result_metrics = {
        "point_top1": float(point_top1),
        "n_labeled": float(n_eval),
        "n_correct": float(n_correct),
        "candidate_miss_rate": float(miss),
        "candidate_miss_n": float(n_miss_eval),
        "label_coverage": float(alignment.coverage),
        "alignment_median_m": float(np.median(alignment.distance_m)) if alignment.distance_m.size else float("nan"),
        "n_matched_observations": float(np.sum(predicted_arcs >= 0)),
        "n_observations": float(predicted_arcs.size),
        **continuity,
        **sequence,
    }
    return EvalResult(
        trip_id=trip.meta.trip_id,
        metrics=result_metrics,
        n_observations=int(predicted_arcs.size),
        n_labeled=int(n_eval),
        n_matched=int(np.sum(predicted_arcs >= 0)),
    )


def align_trip(trip: TripData, max_distance_m: float = config.LABEL_MAX_DISTANCE_M) -> Alignment:
    if trip.route is None:
        raise ValueError(f"Trip {trip.meta.trip_id} has no reference route")
    return labels.align_track_to_route(trip, max_distance_m=max_distance_m)


def load_alignment(trip_dir: str | Path) -> tuple[TripData, Alignment]:
    """Load a trip *with* its reference route and align it (scoring only)."""
    from . import io

    trip = io.load_trip(trip_dir, with_route=True)
    if trip.route is None:
        raise ValueError(f"Trip {trip.meta.trip_id} has no reference route")
    return trip, align_trip(trip)


def polyline_from_arcs(network: Network, arc_ids: np.ndarray) -> list[list[float]]:
    """Convert an arc sequence into a list of ``[lon, lat]`` vertices.

    Consecutive arcs are chained through shared nodes; if they are not
    connected, a new segment is started and the UI will draw a gap.
    """
    coords: list[list[float]] = []
    previous_end: int | None = None
    for arc in metrics.dedupe_sequence(arc_ids):
        from_node, to_node = int(network.arcs[arc, 0]), int(network.arcs[arc, 1])
        if previous_end != from_node:
            if coords:
                # Separate disconnected runs by repeating the last vertex.
                coords.append(list(coords[-1]))
            coords.append([float(network.nodes[from_node, 0]), float(network.nodes[from_node, 1])])
        coords.append([float(network.nodes[to_node, 0]), float(network.nodes[to_node, 1])])
        previous_end = to_node
    return coords
