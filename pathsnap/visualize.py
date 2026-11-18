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
) -> folium.Map:
    lon = trip.track.lonlat[:, 0]
    lat = trip.track.lonlat[:, 1]
    center = (float(lat.mean()), float(lon.mean()))
    fmap = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap", control_scale=True)

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
        ref_layer = folium.FeatureGroup(name="Reference route", show=False)
        for arc in np.asarray(reference_arcs):
            a, b = trip.network.arcs[int(arc)]
            folium.PolyLine(
                locations=[
                    [float(trip.network.nodes[a, 1]), float(trip.network.nodes[a, 0])],
                    [float(trip.network.nodes[b, 1]), float(trip.network.nodes[b, 0])],
                ],
                color="#2e7d32",
                weight=4,
                opacity=0.8,
            ).add_to(ref_layer)
        ref_layer.add_to(fmap)

    if predicted_arcs is not None and len(predicted_arcs):
        pred_layer = folium.FeatureGroup(name="Predicted route", show=True)
        previous_end: int | None = None
        for arc in np.asarray(predicted_arcs):
            arc = int(arc)
            if arc < 0:
                continue
            a, b = trip.network.arcs[arc]
            if previous_end is not None and previous_end != int(a):
                # Disconnected run: keep the visual gap explicit.
                pass
            folium.PolyLine(
                locations=[
                    [float(trip.network.nodes[a, 1]), float(trip.network.nodes[a, 0])],
                    [float(trip.network.nodes[b, 1]), float(trip.network.nodes[b, 0])],
                ],
                color="#c62828",
                weight=4,
                opacity=0.9,
            ).add_to(pred_layer)
            previous_end = int(b)
        pred_layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
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
