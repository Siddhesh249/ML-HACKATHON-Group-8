"""Streamlit evaluator UI — runs on Laptop 2.

This client intentionally imports no model, data loader or reference route. It
only talks HTTP to the FastAPI service on Laptop 1.
"""

from __future__ import annotations

import os

import folium
import requests
import streamlit as st
import streamlit.components.v1 as components

DEFAULT_API = os.environ.get("PATHSNAP_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.environ.get("PATHSNAP_UI_TIMEOUT", "300"))

st.set_page_config(page_title="PathSnap — Group 8", page_icon="🗺️", layout="wide")
st.title("PathSnap — Map-Matching & Route Reconstruction")
st.caption("Group 8 · Laptop 2 client · the model, road network and ground truth stay on Laptop 1")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("API connection")
    api_url = st.text_input("API base URL", value=DEFAULT_API).rstrip("/")
    if st.button("Check health", use_container_width=True):
        st.session_state["health_checked"] = True
    health = None
    if st.session_state.get("health_checked") or True:
        try:
            health = requests.get(f"{api_url}/health", timeout=10).json()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Cannot reach {api_url}: {exc}")
            health = None
    if health:
        if health.get("ready"):
            st.success(f"Model ready · {health.get('model_name')} · {health.get('model_version')}")
        else:
            st.warning(health.get("message") or "Model not ready")
        st.caption(
            f"trips={health.get('n_trips')} · partition={health.get('partition')} "
            f"· features={health.get('feature_version')}"
        )
    st.divider()
    st.markdown(
        "**Ground-truth safety**\n\n"
        "The UI never receives the reference route or per-observation labels. "
        "The API scores the prediction server-side and returns aggregate metrics only."
    )
    st.divider()
    st.caption("Tip: hard-refresh after restarting the API.")

# ---------------------------------------------------------------------------
# Trip selection
# ---------------------------------------------------------------------------
st.subheader("1 · Select an evaluation trip")

trips = []
partition = None
try:
    response = requests.get(f"{api_url}/trips", timeout=30)
    response.raise_for_status()
    payload = response.json()
    trips = payload.get("trips", [])
    partition = payload.get("partition")
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load trips: {exc}")

if not trips:
    st.info("No trips found. Place the evaluator trips under `data/evaluator_test/` on Laptop 1.")
    st.stop()

labels = {
    f"{t['trip_id']}  ·  {t['n_points']} points  ·  {t['n_nodes']} nodes": t["trip_id"]
    for t in trips
}
choice = st.selectbox(f"Available trips ({partition})", list(labels.keys()))
trip_id = labels[choice]
col_a, col_b = st.columns([1, 3])
with col_a:
    mode = st.radio("Decoder", options=["hmm", "classifier"], index=0, horizontal=True)
with col_b:
    st.caption(
        "HMM is the default. Classifier-only is shown for the report comparison; "
        "both are scored by the same server-side reference."
    )
run = st.button("Run prediction", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if run:
    with st.spinner("Running candidates → features → classifier → Viterbi → scoring …"):
        try:
            resp = requests.post(
                f"{api_url}/predict",
                json={"trip_id": trip_id, "mode": mode},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                st.error(f"API {resp.status_code}: {resp.text[:800]}")
                st.stop()
            result = resp.json()
            st.session_state["result"] = result
            st.session_state["result_trip"] = trip_id
        except requests.exceptions.Timeout:
            st.error(f"Prediction timed out after {REQUEST_TIMEOUT:.0f}s. Check the API terminal.")
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")
            st.stop()

result = st.session_state.get("result")
if not result or st.session_state.get("result_trip") != trip_id:
    st.info("Select a trip and press **Run prediction**.")
    st.stop()

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
st.subheader("2 · Server-side evaluation")
metrics = result.get("metrics", {})


def fmt(value, percent=True):
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.2f}%" if percent else f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


cards = st.columns(5)
cards[0].metric("Observation Top-1", fmt(metrics.get("point_top1")))
cards[1].metric("Route continuity", fmt(metrics.get("strict_continuity")))
cards[2].metric("Sequence LCS F1", fmt(metrics.get("lcs_f1")))
cards[3].metric("Arc-set F1", fmt(metrics.get("arc_f1")))
cards[4].metric("Candidate miss", fmt(metrics.get("candidate_miss_rate")))

st.caption(
    f"trip={result['trip_id']} · decoder={result['mode']} · "
    f"model={result['model_name']} {result['model_version']} · "
    f"observations={result['n_observations']} · matched={result['n_matched']} · "
    f"predicted arcs={result['route_arc_count']}"
)

with st.expander("Timings and full metrics", expanded=False):
    timings = result.get("timings_ms", {})
    st.write("**Stage timings (ms)**")
    st.table({k: round(float(v), 1) for k, v in timings.items()})
    st.write("**All metrics**")
    st.json(metrics)

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
st.subheader("3 · Predicted route")
predicted = result.get("predicted_route", [])
observed = result.get("observed_points", [])
if predicted:
    lon = [p[0] for p in predicted]
    lat = [p[1] for p in predicted]
    fmap = folium.Map(location=[sum(lat) / len(lat), sum(lon) / len(lon)], zoom_start=13, control_scale=True)
    observed_layer = folium.FeatureGroup(name="GPS observations", show=True)
    for lon_i, lat_i in observed:
        folium.CircleMarker(
            location=[lat_i, lon_i], radius=2, color="#1565c0",
            fill=True, fill_opacity=0.6, weight=0,
        ).add_to(observed_layer)
    observed_layer.add_to(fmap)
    pred_layer = folium.FeatureGroup(name="Predicted route", show=True)
    folium.PolyLine(
        locations=[[p[1], p[0]] for p in predicted],
        color="#c62828", weight=4, opacity=0.9, tooltip="Predicted route",
    ).add_to(pred_layer)
    pred_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[lat_i, lon_i] for lon_i, lat_i in predicted], padding=(20, 20))
    st.markdown(
        '<div style="font-size:0.9rem;margin:0.25rem 0 0.5rem 0;">'
        '<span style="color:#1565c0;">●</span> GPS observations &nbsp;&nbsp;'
        '<span style="color:#c62828;">━</span> Predicted route'
        '</div>',
        unsafe_allow_html=True,
    )
    components.html(fmap._repr_html_(), height=640)
    st.download_button(
        "Download predicted route (GeoJSON-ish JSON)",
        data=str({"type": "Feature", "geometry": {"type": "LineString", "coordinates": predicted}}),
        file_name=f"pathsnap_{trip_id}.json",
        mime="application/json",
    )
else:
    st.warning("The API returned no predicted route.")

with st.expander("Raw API response", expanded=False):
    st.json(result)
