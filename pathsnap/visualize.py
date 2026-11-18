"""Folium map generation for EDA, the report and the demo.

The evaluation UI deliberately does not import this module's reference-route
helpers: report maps may show the reference route, the live UI may not.
"""

from __future__ import annotations

from pathlib import Path

import folium
import numpy as np

from . import geo
from .contracts import Network, TripData


def _network_arcs_near(
    network: Network,
    bbox: tuple[float, float, float, float],
    margin_deg: float = 0.01,
    max_arcs: int = 6000,
) -> np.ndarray:
    lon_min, lon_max, lat_min, lat_max = bbox
    nodes = network.nodes
    inside = (
        (nodes[:, 0] >= lon_min - margin_deg)
        & (nodes[:, 0] <= lon_max + margin_deg)
        & (nodes[:, 1] >= lat_min - margin_deg)
        & (nodes[:, 1] <= lat_max + margin_deg)
    )
    mask = inside[network.arcs[:, 0]] & inside[network.arcs[:, 1]]
    arcs = np.flatnonzero(mask)
    if arcs.size > max_arcs:
        arcs = arcs[np.linspace(0, arcs.size - 1, max_arcs).astype(int)]
    return arcs


def trip_map(
    trip: TripData,
    predicted_arcs: np.ndarray | None = None,
    reference_arcs: np.ndarray | None = None,
    show_network: bool = True,
    max_observed: int = 4000,
    title: str | None = None,
    show_reference: bool = False,
    legend: bool = True,
    predictions: dict[str, tuple[np.ndarray, str]] | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    tiles: str = "OpenStreetMap",
) -> folium.Map:
    """Folium map of one trip.

    ``predictions`` maps a layer name to ``(arc_ids, colour)`` so several
    decoders (e.g. classifier-only and HMM) can be compared on one map;
    ``predicted_arcs`` is the single-layer shorthand. ``bounds`` =
    ``(lat_min, lon_min, lat_max, lon_max)`` overrides the default fit to the
    whole track (used for zoomed report figures).
    """
    lon = trip.track.lonlat[:, 0]
    lat = trip.track.lonlat[:, 1]
    center = (float(lat.mean()), float(lon.mean()))
    fmap = folium.Map(location=center, zoom_start=13, tiles=tiles, control_scale=True)

    if show_network:
        network_layer = folium.FeatureGroup(name="Road network", show=True)
        for arc in _network_arcs_near(trip.network, trip.meta.bbox):
            a, b = trip.network.arcs[arc]
            folium.PolyLine(
                locations=[
                    [float(trip.network.nodes[a, 1]), float(trip.network.nodes[a, 0])],
                    [float(trip.network.nodes[b, 1]), float(trip.network.nodes[b, 0])],
                ],
                color="#9e9e9e",
                weight=1,
                opacity=0.5,
            ).add_to(network_layer)
        network_layer.add_to(fmap)

    observed_layer = folium.FeatureGroup(name="GPS observations", show=True)
    idx = np.linspace(0, len(lon) - 1, min(max_observed, len(lon))).astype(int)
    for i in idx:
        folium.CircleMarker(
            location=[float(lat[i]), float(lon[i])],
            radius=2,
            color="#1565c0",
            fill=True,
            fill_opacity=0.6,
            weight=0,
            tooltip=f"obs {int(i)} · t={trip.track.t[i]:.0f}s",
        ).add_to(observed_layer)
    observed_layer.add_to(fmap)

    if reference_arcs is not None and len(reference_arcs):
        ref_layer = folium.FeatureGroup(name="Reference route", show=show_reference)
        for arc in np.asarray(reference_arcs):
            a, b = trip.network.arcs[int(arc)]
            folium.PolyLine(
                locations=[
                    [float(trip.network.nodes[a, 1]), float(trip.network.nodes[a, 0])],
                    [float(trip.network.nodes[b, 1]), float(trip.network.nodes[b, 0])],
                ],
                color="#2e7d32",
                weight=9 if predictions else 4,
                opacity=0.45 if predictions else 0.8,
            ).add_to(ref_layer)
        ref_layer.add_to(fmap)

    layers = dict(predictions or {})
    if predicted_arcs is not None and len(predicted_arcs):
        layers.setdefault("Predicted route", (np.asarray(predicted_arcs), "#c62828"))
    # Stacked widths keep every layer visible when routes overlap:
    # wide pale reference underneath, then progressively thinner predictions.
    widths = [5.5, 2.5, 1.5]
    for rank, (name, (arcs_pred, colour)) in enumerate(layers.items()):
        pred_layer = folium.FeatureGroup(name=name, show=True)
        for arc in np.unique(np.asarray(arcs_pred)[np.asarray(arcs_pred) >= 0]):
            a, b = trip.network.arcs[int(arc)]
            folium.PolyLine(
                locations=[
                    [float(trip.network.nodes[a, 1]), float(trip.network.nodes[a, 0])],
                    [float(trip.network.nodes[b, 1]), float(trip.network.nodes[b, 0])],
                ],
                color=colour,
                weight=widths[min(rank, len(widths) - 1)] if len(layers) > 1 else 4,
                opacity=0.95,
                tooltip=name,
            ).add_to(pred_layer)
        pred_layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    if bounds is not None:
        fmap.fit_bounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]])
    else:
        fmap.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    if legend:
        items = [("#1565c0", "GPS observations")]
        if show_network:
            items.insert(0, ("#9e9e9e", "Road network (trip-local)"))
        if reference_arcs is not None and len(reference_arcs):
            items.append(("#2e7d32", "Reference route"))
        items.extend((colour, name) for name, (_, colour) in layers.items())
        rows = "".join(
            f'<div><span style="display:inline-block;width:18px;height:4px;background:{c};'
            f'margin-right:6px;vertical-align:middle"></span>{name}</div>'
            for c, name in items
        )
        legend_html = (
            '<div style="position: fixed; bottom: 24px; left: 12px; z-index: 9999; background: white; '
            'padding: 8px 10px; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,.3); '
            'font-family: sans-serif; font-size: 12px; line-height: 1.6;">' + rows + "</div>"
        )
        fmap.get_root().html.add_child(folium.Element(legend_html))
    if title:
        title_html = (
            '<div style="position: fixed; top: 10px; left: 50px; z-index: 9999; '
            'background: white; padding: 6px 10px; border-radius: 4px; '
            'font-family: sans-serif; font-size: 14px;">' + title + "</div>"
        )
        fmap.get_root().html.add_child(folium.Element(title_html))
    return fmap


def save_map(fmap: folium.Map, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(path))
    return path


def trajectory_overlay(trip: TripData, predicted_arcs: np.ndarray, path: str | Path) -> Path:
    """Convenience wrapper used by the report scripts."""
    return save_map(
        trip_map(
            trip,
            predicted_arcs=predicted_arcs,
            reference_arcs=trip.route,
            title=f"Trip {trip.meta.trip_id}: GPS vs predicted route",
        ),
        path,
    )
