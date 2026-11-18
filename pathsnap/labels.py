"""Reference-route alignment and per-observation labels.

The reference ``.route`` is an arc sequence, not a per-observation label. We
project the track onto the route polyline, enforce monotonic progress, and
assign each observation the arc under its projection. Observations far from the
assigned segment are marked low-confidence and excluded from training.
"""

from __future__ import annotations

import numpy as np

from . import config, geo
from .contracts import Alignment, TripData

BACKWARD_EPS_M = 1e-6


def route_polyline(trip: TripData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Return projected route segments and cumulative arclength.

    Returns
    -------
    a_xy, b_xy : (R, 2)
    lengths_m : (R,)
    arc_ids : (R,)
    cumulative : (R + 1,)
    origin : (lon0, lat0)
    """
    if trip.route is None:
        raise ValueError("Trip has no reference route")
    origin = geo.local_origin(trip.track.lonlat)
    nodes_xy, _ = geo.project_lonlat(trip.network.nodes, origin=origin)
    arcs = trip.network.arcs[trip.route]
    a_xy = nodes_xy[arcs[:, 0]]
    b_xy = nodes_xy[arcs[:, 1]]
    lengths = np.linalg.norm(b_xy - a_xy, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    return a_xy, b_xy, lengths, trip.route.copy(), cumulative, origin


def project_onto_route(trip: TripData, chunk_size: int = 512) -> dict:
    """Nearest route segment, raw route progress and distance per observation.

    ``progress_m`` is *not* made monotonic: it is the arclength position of the
    nearest route point, used by the alignment and by EDA diagnostics
    (backward jumps, reversed route order).
    """
    a_xy, b_xy, lengths, route_arcs, cumulative, origin = route_polyline(trip)
    if lengths.size == 0 or cumulative[-1] <= 0:
        raise ValueError("Reference route has zero length")

    xy, _ = geo.project_lonlat(trip.track.lonlat, origin=origin)
    n = xy.shape[0]
    nearest_seg = np.empty(n, dtype=np.int64)
    s_raw = np.empty(n, dtype=float)
    dist_raw = np.empty(n, dtype=float)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        dist, ratio, _ = geo.point_segment_distance(xy[start:stop], a_xy, b_xy, chunk_size=stop - start)
        idx = np.argmin(dist, axis=1)
        nearest_seg[start:stop] = idx
        s_raw[start:stop] = cumulative[idx] + ratio[np.arange(stop - start), idx] * lengths[idx]
        dist_raw[start:stop] = dist[np.arange(stop - start), idx]
    return {
        "nearest_seg": nearest_seg,
        "progress_m": s_raw,
        "distance_m": dist_raw,
        "route_arcs": route_arcs,
        "route_length_m": float(cumulative[-1]),
    }


def route_order_diagnostics(progress_m: np.ndarray, jump_m: float = 100.0) -> dict:
    """Loop / reversed-route diagnostics from raw route progress.

    ``order_corr`` is the Spearman rank correlation between observation index
    and route progress (+1 = track follows the route order, < 0 = route stored
    in the opposite direction). ``backward_jumps`` counts steps that move more
    than ``jump_m`` backwards along the route, which jitter at a stop cannot do.
    """
    s = np.asarray(progress_m, dtype=float)
    if s.size < 3:
        return {"order_corr": float("nan"), "backward_jumps": 0, "backward_jump_m_max": 0.0}
    rank = np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float)
    idx = np.arange(s.size, dtype=float)
    corr = float(np.corrcoef(idx, rank)[0, 1]) if rank.std() > 0 else float("nan")
    ds = np.diff(s)
    return {
        "order_corr": corr,
        "backward_jumps": int((ds < -jump_m).sum()),
        "backward_jump_m_max": float(max(0.0, -ds.min())),
    }


def align_track_to_route(
    trip: TripData,
    max_distance_m: float = config.LABEL_MAX_DISTANCE_M,
    chunk_size: int = 512,
) -> Alignment:
    """Assign one reference arc per GPS observation."""
    proj = project_onto_route(trip, chunk_size=chunk_size)
    nearest_seg = proj["nearest_seg"]
    s_raw = proj["progress_m"]
    dist_raw = proj["distance_m"]
    route_arcs = proj["route_arcs"]
    n = s_raw.size

    # The vehicle may stop, but it cannot teleport backwards along the route.
    # Backwards movement is a diagnostic of loop/reversed reference routes, not
    # a reason to discard a point: for a per-observation label we only need the
    # arc under the GPS, and the nearest route segment gives that directly.
    if n > 1:
        backward_fraction = float(np.mean(np.diff(s_raw) < -BACKWARD_EPS_M))
    else:
        backward_fraction = 0.0
    s_mono = np.maximum.accumulate(s_raw)

    dist_assigned = dist_raw
    confidence = dist_assigned <= float(max_distance_m)

    labels = np.full(n, -1, dtype=np.int64)
    labels[confidence] = route_arcs[nearest_seg[confidence]]

    return Alignment(
        arc_ids=labels,
        confidence=confidence,
        distance_m=dist_assigned,
        progress_m=s_mono,
        route_length_m=proj["route_length_m"],
        coverage=float(confidence.mean()) if n else 0.0,
        backward_fraction=backward_fraction,
    )


def alignment_stats(alignment: Alignment) -> dict:
    """Small summary used in EDA and the report."""
    conf = alignment.confidence
    return {
        "n_observations": int(conf.size),
        "n_confident": int(conf.sum()),
        "coverage": alignment.coverage,
        "median_distance_m": float(np.median(alignment.distance_m)) if conf.size else float("nan"),
        "p90_distance_m": float(np.percentile(alignment.distance_m, 90)) if conf.size else float("nan"),
        "route_length_m": alignment.route_length_m,
        "backward_fraction": alignment.backward_fraction,
        "progress_m": float(alignment.progress_m[-1]) if conf.size else 0.0,
    }
