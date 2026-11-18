"""Geographic helpers: local projection, distances, bearings, smoothing."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8
M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON_AT_EQUATOR = 111_320.0


def local_origin(lonlat: np.ndarray) -> tuple[float, float]:
    """Return a stable local origin (mean longitude, mean latitude)."""
    arr = np.asarray(lonlat, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def project_lonlat(
    lonlat: np.ndarray,
    origin: tuple[float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Project (lon, lat) to local metres with an equirectangular approximation.

    The error is negligible at trip scale (a few tens of kilometres) and the
    projection keeps every geometric operation vectorised and fast.
    """
    arr = np.atleast_2d(np.asarray(lonlat, dtype=float))
    if origin is None:
        origin = local_origin(arr)
    lon0, lat0 = origin
    mx = M_PER_DEG_LON_AT_EQUATOR * np.cos(np.deg2rad(lat0))
    my = M_PER_DEG_LAT
    xy = np.column_stack(((arr[:, 0] - lon0) * mx, (arr[:, 1] - lat0) * my))
    return xy, origin


def unproject_xy(xy: np.ndarray, origin: tuple[float, float]) -> np.ndarray:
    """Inverse of :func:`project_lonlat`."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    lon0, lat0 = origin
    mx = M_PER_DEG_LON_AT_EQUATOR * np.cos(np.deg2rad(lat0))
    my = M_PER_DEG_LAT
    return np.column_stack((xy[:, 0] / mx + lon0, xy[:, 1] / my + lat0))


def haversine_m(
    lon1: np.ndarray | float,
    lat1: np.ndarray | float,
    lon2: np.ndarray | float,
    lat2: np.ndarray | float,
) -> np.ndarray:
    """Great-circle distance in metres (vectorised)."""
    lon1, lat1, lon2, lat2 = map(
        np.radians, (np.asarray(lon1), np.asarray(lat1), np.asarray(lon2), np.asarray(lat2))
    )
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def point_segment_distance(
    points: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distance from each point to each segment.

    Parameters
    ----------
    points : (N, 2)
    seg_a, seg_b : (M, 2)

    Returns
    -------
    dist : (N, M)
    ratio : (N, M) in [0, 1]
    projection : (N, M, 2)
    """
    p = np.atleast_2d(np.asarray(points, dtype=float))
    a = np.atleast_2d(np.asarray(seg_a, dtype=float))
    b = np.atleast_2d(np.asarray(seg_b, dtype=float))
    ab = b - a
    denom = np.einsum("mi,mi->m", ab, ab)
    safe_denom = np.where(denom > 1e-12, denom, 1.0)

    n, m = p.shape[0], a.shape[0]
    dist = np.empty((n, m), dtype=np.float64)
    ratio = np.empty((n, m), dtype=np.float64)
    projection = np.empty((n, m, 2), dtype=np.float64)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        pp = p[start:stop, None, :]
        ap = pp - a[None, :, :]
        t = np.einsum("nmi,mi->nm", ap, ab) / safe_denom[None, :]
        t = np.clip(t, 0.0, 1.0)
        proj = a[None, :, :] + t[..., None] * ab[None, :, :]
        dist[start:stop] = np.linalg.norm(pp - proj, axis=-1)
        ratio[start:stop] = t
        projection[start:stop] = proj
    return dist, ratio, projection


def distance_to_assigned_segments(
    points: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    assigned: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance/ratio from each point to its own assigned segment."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    a = np.atleast_2d(np.asarray(seg_a, dtype=float))[assigned]
    b = np.atleast_2d(np.asarray(seg_b, dtype=float))[assigned]
    ab = b - a
    ap = p - a
    denom = np.einsum("ni,ni->n", ab, ab)
    safe = np.where(denom > 1e-12, denom, 1.0)
    t = np.clip(np.einsum("ni,ni->n", ap, ab) / safe, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=1), t


def bearing_deg(
    lon1: np.ndarray | float,
    lat1: np.ndarray | float,
    lon2: np.ndarray | float,
    lat2: np.ndarray | float,
) -> np.ndarray:
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    lon1, lat1, lon2, lat2 = map(
        np.radians, (np.asarray(lon1), np.asarray(lat1), np.asarray(lon2), np.asarray(lat2))
    )
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def wrap_angle_deg(angle: np.ndarray | float) -> np.ndarray:
    """Wrap an angle to (-180, 180]."""
    a = np.asarray(angle, dtype=float)
    return (a + 180.0) % 360.0 - 180.0


def angle_difference_deg(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Signed smallest difference a - b in degrees."""
    return wrap_angle_deg(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def smooth_track(
    lonlat: np.ndarray,
    t: np.ndarray,
    window: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth GPS positions and derive heading (deg) and speed (m/s).

    Returns
    -------
    smooth_lonlat : (N, 2)
    heading_deg : (N,)
    speed_mps : (N,)
    """
    lonlat = np.atleast_2d(np.asarray(lonlat, dtype=float))
    t = np.asarray(t, dtype=float)
    n = lonlat.shape[0]
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if n == 0:
        return lonlat.copy(), np.zeros(0), np.zeros(0)
    if window == 1:
        smooth = lonlat.copy()
    else:
        half = window // 2
        padded = np.pad(lonlat, ((half, half), (0, 0)), mode="edge")
        kernel = np.ones(window) / window
        smooth = np.column_stack(
            (
                np.convolve(padded[:, 0], kernel, mode="valid"),
                np.convolve(padded[:, 1], kernel, mode="valid"),
            )
        )
        smooth = smooth[:n]

    # Headings from a centred difference, with edge handling.
    if n == 1:
        heading = np.zeros(1)
        speed = np.zeros(1)
    else:
        prev_idx = np.maximum(np.arange(n) - 1, 0)
        next_idx = np.minimum(np.arange(n) + 1, n - 1)
        heading = bearing_deg(
            smooth[prev_idx, 0],
            smooth[prev_idx, 1],
            smooth[next_idx, 0],
            smooth[next_idx, 1],
        )
        xy, _ = project_lonlat(smooth)
        step = np.linalg.norm(xy[next_idx] - xy[prev_idx], axis=1)
        span = np.maximum(t[next_idx] - t[prev_idx], 0.0)
        speed = np.divide(step, span, out=np.zeros_like(step), where=span > 0)
    return smooth, heading, speed


def cumulative_lengths(points_xy: np.ndarray) -> np.ndarray:
    """Cumulative arclength at each vertex, starting at zero."""
    pts = np.atleast_2d(np.asarray(points_xy, dtype=float))
    if pts.shape[0] == 0:
        return np.zeros(0)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(steps)))
