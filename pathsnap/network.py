"""Road-network utilities: geometry, topology and corridor subgraphs."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from . import geo
from .contracts import Network


def projected_nodes(network: Network, origin: tuple[float, float] | None = None):
    return geo.project_lonlat(network.nodes, origin=origin)


def arc_geometry(network: Network, origin: tuple[float, float] | None = None):
    """Return projected arc endpoints, lengths and headings.

    Returns
    -------
    a_xy, b_xy : (A, 2)
    lengths_m : (A,)
    heading_deg : (A,) bearings from ``from_node`` to ``to_node``
    origin : (lon0, lat0)
    """
    nodes_xy, used_origin = geo.project_lonlat(network.nodes, origin=origin)
    a_xy = nodes_xy[network.arcs[:, 0]]
    b_xy = nodes_xy[network.arcs[:, 1]]
    lengths = np.linalg.norm(b_xy - a_xy, axis=1)
    heading = np.degrees(np.arctan2(b_xy[:, 0] - a_xy[:, 0], b_xy[:, 1] - a_xy[:, 1])) % 360.0
    return a_xy, b_xy, lengths, heading, used_origin


def node_degrees(network: Network) -> tuple[np.ndarray, np.ndarray]:
    out_deg = np.bincount(network.arcs[:, 0], minlength=network.n_nodes)
    in_deg = np.bincount(network.arcs[:, 1], minlength=network.n_nodes)
    return out_deg, in_deg


def reciprocal_mask(network: Network) -> np.ndarray:
    """True where the reverse arc (to -> from) also exists."""
    codes = network.arcs[:, 0].astype(np.int64) * network.n_nodes + network.arcs[:, 1]
    reverse = network.arcs[:, 1].astype(np.int64) * network.n_nodes + network.arcs[:, 0]
    present = np.isin(reverse, codes)
    return present


def components(network: Network) -> tuple[int, np.ndarray]:
    n = network.n_nodes
    graph = csr_matrix(
        (np.ones(network.n_arcs), (network.arcs[:, 0], network.arcs[:, 1])),
        shape=(n, n),
    )
    return connected_components(graph, directed=False)


def topology_stats(network: Network) -> dict:
    out_deg, in_deg = node_degrees(network)
    n_comp, labels = components(network)
    sizes = np.bincount(labels)
    return {
        "n_nodes": network.n_nodes,
        "n_arcs": network.n_arcs,
        "n_components": int(n_comp),
        "largest_component": int(sizes.max()) if sizes.size else 0,
        "isolated_nodes": int((sizes == 1).sum()) if sizes.size else 0,
        "max_out_degree": int(out_deg.max()) if out_deg.size else 0,
        "max_in_degree": int(in_deg.max()) if in_deg.size else 0,
    }


def build_csr(
    network: Network,
    arc_mask: np.ndarray | None = None,
    origin: tuple[float, float] | None = None,
) -> tuple[csr_matrix, np.ndarray, tuple[float, float]]:
    """Build a directed weighted CSR graph (weights = arc length in metres)."""
    _, _, lengths, _, used_origin = arc_geometry(network, origin=origin)
    if arc_mask is None:
        arc_mask = np.ones(network.n_arcs, dtype=bool)
    arcs = network.arcs[arc_mask]
    weights = np.maximum(lengths[arc_mask], 1e-3)
    graph = csr_matrix(
        (weights, (arcs[:, 0], arcs[:, 1])),
        shape=(network.n_nodes, network.n_nodes),
    )
    return graph, lengths, used_origin


def corridor_subgraph(
    network: Network,
    points_xy: np.ndarray,
    origin: tuple[float, float] | None = None,
    margin_m: float = 400.0,
) -> dict:
    """Restrict the network to nodes near the GPS trace.

    The corridor keeps shortest-path queries cheap and bounded. We select nodes
    within ``margin_m`` of any observation, then add both endpoints of every
    incident arc so local connectivity is preserved.
    """
    nodes_xy, used_origin = geo.project_lonlat(network.nodes, origin=origin)
    points = np.atleast_2d(np.asarray(points_xy, dtype=float))
    tree = cKDTree(nodes_xy)
    near = tree.query_ball_point(points, r=margin_m)
    selected = np.zeros(network.n_nodes, dtype=bool)
    if len(near) > 0:
        idx = np.concatenate([np.asarray(x, dtype=np.int64) for x in near if len(x)]) if any(len(x) for x in near) else np.empty(0, dtype=np.int64)
        if idx.size:
            selected[idx] = True

    # Add endpoints of incident arcs (one pass) to avoid cutting the corridor.
    incident = selected[network.arcs[:, 0]] | selected[network.arcs[:, 1]]
    selected[network.arcs[incident, 0]] = True
    selected[network.arcs[incident, 1]] = True

    local_index = np.full(network.n_nodes, -1, dtype=np.int64)
    global_ids = np.flatnonzero(selected)
    local_index[global_ids] = np.arange(global_ids.size)

    arc_mask = selected[network.arcs[:, 0]] & selected[network.arcs[:, 1]]
    _, _, lengths, _, _ = arc_geometry(network, origin=used_origin)
    arcs = network.arcs[arc_mask]
    weights = np.maximum(lengths[arc_mask], 1e-3)
    graph = csr_matrix(
        (weights, (local_index[arcs[:, 0]], local_index[arcs[:, 1]])),
        shape=(global_ids.size, global_ids.size),
    )
    return {
        "graph": graph,
        "local_index": local_index,
        "global_ids": global_ids,
        "arc_mask": arc_mask,
        "origin": used_origin,
        "node_xy": nodes_xy,
        "arc_lengths_m": lengths,
    }


def shortest_distances(
    graph: csr_matrix,
    source: int,
    limit_m: float,
) -> np.ndarray:
    """Directed shortest-path distances from one node, capped at ``limit_m``."""
    if source < 0 or source >= graph.shape[0]:
        return np.full(graph.shape[1], np.inf)
    return dijkstra(graph, directed=True, indices=source, limit=limit_m)


def distance_between_nodes(
    graph: csr_matrix,
    source: int,
    target: int,
    limit_m: float,
) -> float:
    if source < 0 or target < 0:
        return float("inf")
    row = shortest_distances(graph, source, limit_m)
    value = row[target] if target < row.size else np.inf
    return float(value)
