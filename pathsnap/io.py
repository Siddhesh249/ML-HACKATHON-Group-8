"""Trip discovery, loading and validation.

The prediction path must call ``load_trip(..., with_route=False)``. Only the
scoring path may load ``.route``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import config
from .contracts import DataError, Network, Track, TripData, TripMeta

REQUIRED_SUFFIXES = (".track", ".nodes", ".arcs")
OPTIONAL_SUFFIXES = (".route",)


def _read_matrix(path: Path, ncols: int, dtype=float) -> np.ndarray:
    if not path.exists():
        raise DataError(f"Missing file: {path}")
    try:
        arr = np.atleast_2d(np.loadtxt(path, dtype=dtype, ndmin=2))
    except Exception as exc:  # noqa: BLE001 - report the offending file
        raise DataError(f"Could not parse {path}: {exc}") from exc
    if arr.size == 0:
        return np.empty((0, ncols), dtype=dtype)
    if arr.shape[1] != ncols:
        raise DataError(f"{path}: expected {ncols} columns, found {arr.shape[1]}")
    return arr


def load_track(path: Path) -> Track:
    arr = _read_matrix(path, 3)
    if arr.shape[0] == 0:
        raise DataError(f"{path}: empty track")
    lonlat = arr[:, :2].astype(float)
    t = arr[:, 2].astype(float)
    if not np.isfinite(lonlat).all() or not np.isfinite(t).all():
        raise DataError(f"{path}: non-finite values")
    if np.any(np.abs(lonlat[:, 0]) > 180) or np.any(np.abs(lonlat[:, 1]) > 90):
        raise DataError(f"{path}: coordinates outside lon/lat range")
    return Track(lonlat=lonlat, t=t)


def load_network(nodes_path: Path, arcs_path: Path) -> Network:
    nodes = _read_matrix(nodes_path, 2)
    arcs = _read_matrix(arcs_path, 2, dtype=np.int64)
    if nodes.shape[0] == 0:
        raise DataError(f"{nodes_path}: empty node file")
    if arcs.shape[0] == 0:
        raise DataError(f"{arcs_path}: empty arc file")
    if arcs.min() < 0 or arcs.max() >= nodes.shape[0]:
        raise DataError(f"{arcs_path}: arc endpoint outside node range")
    if not np.isfinite(nodes).all():
        raise DataError(f"{nodes_path}: non-finite coordinates")
    return Network(nodes=nodes, arcs=arcs)


def load_route(path: Path, n_arcs: int) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.atleast_2d(np.loadtxt(path, dtype=np.int64, ndmin=2))
    if arr.size == 0:
        return None
    route = arr[:, 0] if arr.ndim == 2 else arr
    if route.min() < 0 or route.max() >= n_arcs:
        raise DataError(f"{path}: route arc ID outside arc range")
    return route.astype(np.int64)


def trip_meta(directory: Path, partition: str) -> TripMeta:
    trip_id = directory.name
    track = load_track(directory / f"{trip_id}.track")
    network = load_network(
        directory / f"{trip_id}.nodes",
        directory / f"{trip_id}.arcs",
    )
    route_path = directory / f"{trip_id}.route"
    has_route = route_path.exists()
    n_route = 0
    if has_route:
        route = load_route(route_path, network.n_arcs)
        n_route = 0 if route is None else int(route.shape[0])
    lon = track.lonlat[:, 0]
    lat = track.lonlat[:, 1]
    return TripMeta(
        trip_id=trip_id,
        partition=partition,
        directory=directory,
        n_points=len(track),
        n_nodes=network.n_nodes,
        n_arcs=network.n_arcs,
        n_route_arcs=n_route,
        has_route=has_route,
        bbox=(float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())),
    )


def load_trip(directory: str | Path, with_route: bool = True) -> TripData:
    directory = Path(directory)
    if not directory.is_dir():
        raise DataError(f"Not a trip directory: {directory}")
    meta = trip_meta(directory, partition=directory.parent.name)
    track = load_track(directory / f"{meta.trip_id}.track")
    network = load_network(
        directory / f"{meta.trip_id}.nodes",
        directory / f"{meta.trip_id}.arcs",
    )
    route = None
    if with_route:
        route = load_route(directory / f"{meta.trip_id}.route", network.n_arcs)
    return TripData(meta=meta, track=track, network=network, route=route)


def discover_trips(
    data_root: str | Path | None = None,
    partition: str | None = None,
) -> list[TripMeta]:
    """Discover trip directories under ``<data_root>/<partition>``."""
    root = Path(data_root) if data_root is not None else config.DATA_ROOT
    partitions = [partition] if partition else [
        p.name for p in sorted(root.glob("*")) if p.is_dir()
    ]
    metas: list[TripMeta] = []
    for part in partitions:
        part_dir = root / part
        if not part_dir.is_dir():
            continue
        for trip_dir in sorted(part_dir.iterdir()):
            if not trip_dir.is_dir():
                continue
            trip_id = trip_dir.name
            if not all((trip_dir / f"{trip_id}{suffix}").exists() for suffix in REQUIRED_SUFFIXES):
                continue
            try:
                metas.append(trip_meta(trip_dir, part))
            except DataError:
                # A malformed trip should not take down the whole discovery.
                continue
    return metas


def resolve_trip_dir(trip_id: str, data_root: str | Path | None = None) -> Path:
    """Resolve a trip ID to a directory, preferring evaluator trips."""
    import re

    if not re.fullmatch(r"[0-9]{8}", trip_id):
        raise DataError("trip_id must be exactly eight digits")
    root = Path(data_root) if data_root is not None else config.DATA_ROOT
    for partition in (config.EVALUATION_PARTITION, config.FALLBACK_PARTITION):
        candidate = root / partition / trip_id
        if candidate.is_dir():
            return candidate
    for candidate in sorted(root.glob(f"*/{trip_id}")):
        if candidate.is_dir():
            return candidate
    raise DataError(f"Trip not found under {root}: {trip_id}")


def trip_fingerprint(directory: Path) -> str:
    """Stable hash of the raw files, used in experiment logs."""
    digest = hashlib.sha256()
    for suffix in REQUIRED_SUFFIXES + OPTIONAL_SUFFIXES:
        path = directory / f"{directory.name}{suffix}"
        if path.exists():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def inventory_rows(data_root: str | Path | None = None) -> list[dict]:
    rows = []
    for meta in discover_trips(data_root):
        rows.append(meta.to_dict())
    return rows
