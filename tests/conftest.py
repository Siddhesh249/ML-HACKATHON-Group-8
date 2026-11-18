"""Fixture: a small synthetic trip so tests never require the real dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _build_grid():
    nodes = []
    for row in range(3):
        for col in range(3):
            nodes.append((8.0 + 0.001 * col, 49.0 + 0.001 * row))
    arcs = []

    def add(u: int, v: int) -> None:
        arcs.append((u, v))

    for row in range(3):
        for col in range(3):
            node = row * 3 + col
            if col < 2:
                add(node, node + 1)
                add(node + 1, node)
            if row < 2:
                add(node, node + 3)
                add(node + 3, node)
    return np.asarray(nodes, dtype=float), np.asarray(arcs, dtype=np.int64)


@pytest.fixture()
def synthetic_root(tmp_path: Path):
    nodes, arcs = _build_grid()
    trip_id = "00000001"
    trip_dir = tmp_path / "student_test" / trip_id
    trip_dir.mkdir(parents=True)

    rng = np.random.default_rng(7)
    n = 50
    lon = np.linspace(8.00015, 8.00185, n)
    lat = np.full(n, 49.001) + rng.normal(0.0, 8e-6, size=n)
    t = np.arange(1, n + 1, dtype=float)
    track = np.column_stack([lon, lat, t])

    # Reference route: node 3 -> node 4 -> node 5 along the middle row.
    route = np.asarray(
        [
            int(np.flatnonzero((arcs[:, 0] == 3) & (arcs[:, 1] == 4))[0]),
            int(np.flatnonzero((arcs[:, 0] == 4) & (arcs[:, 1] == 5))[0]),
        ],
        dtype=np.int64,
    )

    np.savetxt(trip_dir / f"{trip_id}.track", track, fmt="%.9f")
    np.savetxt(trip_dir / f"{trip_id}.nodes", nodes, fmt="%.9f")
    np.savetxt(trip_dir / f"{trip_id}.arcs", arcs, fmt="%d")
    np.savetxt(trip_dir / f"{trip_id}.route", route, fmt="%d")
    return tmp_path, trip_id, route
