"""Streamlit evaluator UI — runs on Laptop 2.

This client intentionally imports no model, data loader or reference route. It
only talks HTTP to the FastAPI service on Laptop 1.
"""

from __future__ import annotations

import json
import os

import folium
import requests
import streamlit as st

DEFAULT_API = os.environ.get("PATHSNAP_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.environ.get("PATHSNAP_UI_TIMEOUT", "300"))

st.set_page_config(
    page_title="PathSnap — Group 8",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --ps-bg: #0d1117;
        --ps-panel: #151b23;
        --ps-panel-2: #1b2230;
        --ps-border: #263043;
        --ps-text: #e6edf3;
        --ps-muted: #93a1b5;
        --ps-accent: #7c9cff;
        --ps-ok: #3fb950;
        --ps-warn: #d29922;
        --ps-bad: #f85149;
        --ps-gps: #58a6ff;
        --ps-route: #ff7b72;
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }
    section[data-testid="stSidebar"] {
        background: #0f1520;
        border-right: 1px solid var(--ps-border);
    }
    .ps-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
        padding: 1.05rem 1.25rem;
        border: 1px solid var(--ps-border);
        border-radius: 16px;
        background: linear-gradient(135deg, #141c27 0%, #0f1520 100%);
        margin-bottom: 1.1rem;
    }
    .ps-brand { display: flex; align-items: center; gap: .8rem; }
    .ps-logo {
        width: 2.5rem; height: 2.5rem; border-radius: 12px;
        display: grid; place-items: center; font-size: 1.35rem;
        background: rgba(124, 156, 255, 0.14);
        border: 1px solid rgba(124, 156, 255, 0.28);
    }
    .ps-title { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.01em; }
    .ps-sub { font-size: .8rem; color: var(--ps-muted); margin-top: .1rem; }
    .ps-status { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }
    .ps-pill {
        display: inline-flex; align-items: center; gap: .35rem;
        padding: .25rem .6rem; border-radius: 999px;
        font-size: .72rem; font-weight: 600; letter-spacing: .02em;
        border: 1px solid transparent;
    }
    .ps-pill-ok {
        color: var(--ps-ok);
        background: rgba(63, 185, 80, 0.1);
        border-color: rgba(63, 185, 80, 0.3);
    }
    .ps-pill-warn {
        color: var(--ps-warn);
        background: rgba(210, 153, 34, 0.1);
        border-color: rgba(210, 153, 34, 0.3);
    }
    .ps-pill-bad {
        color: var(--ps-bad);
        background: rgba(248, 81, 73, 0.1);
        border-color: rgba(248, 81, 73, 0.3);
    }
    .ps-meta {
        font-size: .72rem; color: var(--ps-muted);
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .ps-step {
        display: flex; align-items: center; gap: .55rem;
        font-size: .82rem; color: var(--ps-muted);
        margin: .15rem 0 .55rem;
    }
    .ps-step-num {
        width: 1.4rem; height: 1.4rem; border-radius: 50%;
        background: var(--ps-accent); color: #0d1117;
        display: grid; place-items: center;
        font-weight: 700; font-size: .72rem;
    }
    .ps-card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
        gap: .7rem;
        margin: .3rem 0 .8rem;
    }
    .ps-card {
        background: var(--ps-panel);
        border: 1px solid var(--ps-border);
        border-radius: 14px;
        padding: .85rem 1rem;
    }
    .ps-card-label {
        font-size: .68rem; text-transform: uppercase; letter-spacing: .09em;
        color: var(--ps-muted); font-weight: 600;
    }
    .ps-card-value {
        font-size: 1.5rem; font-weight: 650; line-height: 1.3;
        margin: .15rem 0 .1rem; font-variant-numeric: tabular-nums;
    }
    .ps-card-hint { font-size: .7rem; color: var(--ps-muted); }
    .ps-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .1rem 0 .6rem; }
    .ps-chip {
        font-size: .7rem; color: var(--ps-muted);
        background: var(--ps-panel-2);
        border: 1px solid var(--ps-border);
        border-radius: 999px; padding: .2rem .55rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .ps-flow { display: flex; flex-wrap: wrap; align-items: center; gap: .35rem; margin: .2rem 0 .5rem; }
    .ps-flow span {
        font-size: .72rem; color: var(--ps-text);
        background: var(--ps-panel-2);
        border: 1px solid var(--ps-border);
        border-radius: 8px; padding: .28rem .55rem;
    }
    .ps-flow em { color: var(--ps-muted); font-style: normal; font-size: .8rem; }
    .ps-callout {
        border: 1px solid rgba(210, 153, 34, 0.32);
        background: rgba(210, 153, 34, 0.08);
        border-radius: 12px;
        padding: .7rem .9rem;
        font-size: .8rem;
        color: #e8d3a4;
        margin: .4rem 0 .8rem;
    }
    .ps-legend { display: flex; gap: 1rem; font-size: .78rem; color: var(--ps-muted); margin: .35rem 0 .4rem; }
    .ps-dot-blue { color: var(--ps-gps); }
    .ps-dot-red { color: var(--ps-route); }
    div[data-testid="stMetric"] {
        background: var(--ps-panel);
        border: 1px solid var(--ps-border);
        border-radius: 14px;
        padding: .7rem .9rem;
    }
    div.stButton > button[kind="primary"] {
        border-radius: 10px;
        font-weight: 600;
        background: linear-gradient(135deg, #7c9cff 0%, #4f6ef7 100%);
        border: 1px solid #8aa4ff;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #8aa4ff 0%, #5f7cff 100%);
        border-color: #9db3ff;
    }
    div[data-testid="stIFrame"] iframe,
    iframe[data-testid="stIFrame"] {
        border-radius: 14px;
        border: 1px solid var(--ps-border);
    }
    div.stButton > button { border-radius: 10px; }
    details[data-testid="stExpander"] {
        border: 1px solid var(--ps-border);
        border-radius: 12px;
        background: var(--ps-panel);
    }
    /* Hide Streamlit chrome so the demo looks like a product, not an editor.
       stHeader is fixed-position and was clipping the top of the header card. */
    #MainMenu, footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"], header[data-testid="stHeader"] {
        display: none !important;
    }
    .ps-state {
        border-radius: 12px;
        padding: .6rem .75rem;
        font-size: .78rem;
        font-weight: 600;
        border: 1px solid transparent;
        margin-bottom: .35rem;
    }
    .ps-state-ok {
        color: #7ee787;
        background: rgba(63, 185, 80, 0.08);
        border-color: rgba(63, 185, 80, 0.28);
    }
    .ps-state-warn {
        color: #e3b341;
        background: rgba(210, 153, 34, 0.08);
        border-color: rgba(210, 153, 34, 0.28);
    }
    .ps-state-bad {
        color: #ff7b72;
        background: rgba(248, 81, 73, 0.08);
        border-color: rgba(248, 81, 73, 0.28);
    }
    .ps-callout-muted {
        border: 1px solid var(--ps-border);
        background: var(--ps-panel);
        border-radius: 12px;
        padding: .8rem 1rem;
        font-size: .82rem;
        color: var(--ps-muted);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def fetch_health(api_url: str) -> dict | None:
    """Return /health, or None when the API is unreachable."""
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:  # noqa: BLE001
        return None


def fetch_trips(api_url: str) -> tuple[dict | None, str | None]:
    try:
        response = requests.get(f"{api_url}/trips", timeout=30)
        response.raise_for_status()
        payload = response.json()
        return payload, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def run_prediction(api_url: str, trip_id: str, mode: str) -> dict:
    response = requests.post(
        f"{api_url}/predict",
        json={"trip_id": trip_id, "mode": mode},
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(f"API {response.status_code}: {response.text[:600]}")
    return response.json()


def fmt_pct(value) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def metric_card(label: str, value: str, hint: str) -> str:
    return (
        '<div class="ps-card">'
        f'<div class="ps-card-label">{label}</div>'
        f'<div class="ps-card-value">{value}</div>'
        f'<div class="ps-card-hint">{hint}</div>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### API connection")
    api_url = st.text_input("Base URL", value=DEFAULT_API, label_visibility="collapsed")
    api_url = api_url.rstrip("/")
    if st.button("Reconnect", use_container_width=True):
        st.rerun()

    health = fetch_health(api_url)
    if health is None:
        st.markdown(
            '<div class="ps-state ps-state-bad">● API unreachable</div>',
            unsafe_allow_html=True,
        )
        st.caption("Start Laptop 1: `uvicorn backend.app:app --host 0.0.0.0 --port 8000`")
    elif health.get("ready"):
        st.markdown(
            '<div class="ps-state ps-state-ok">● Model ready</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"{health.get('model_name')} · `{health.get('model_version')}`  \n"
            f"trips {health.get('n_trips')} · partition {health.get('partition')} · "
            f"features {health.get('feature_version')}"
        )
    else:
        st.markdown(
            '<div class="ps-state ps-state-warn">● Model not ready</div>',
            unsafe_allow_html=True,
        )
        if health.get("message"):
            st.caption(health["message"])

    st.divider()
    st.markdown("#### Pipeline")
    st.markdown(
        '<div class="ps-flow">'
        "<span>GPS</span><em>→</em><span>Candidates</span><em>→</em>"
        "<span>Features</span><em>→</em><span>Classifier</span><em>→</em>"
        "<span>HMM</span><em>→</em><span>Score</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption("Candidates and scoring run on Laptop 1. The client only renders the result.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
ready = bool(health and health.get("ready"))
partition = health.get("partition") if health else None
if ready:
    status_html = '<span class="ps-pill ps-pill-ok">● Ready</span>'
elif health is not None:
    status_html = '<span class="ps-pill ps-pill-warn">● Model not ready</span>'
else:
    status_html = '<span class="ps-pill ps-pill-bad">● API offline</span>'

meta_html = ""
if health:
    meta_html = (
        f'<span class="ps-meta">{health.get("model_name")} · {health.get("model_version")}'
        f" · {health.get('n_trips')} trips</span>"
    )
fallback_html = ""
if partition and partition != "evaluator_test":
    fallback_html = '<span class="ps-pill ps-pill-warn">student-test fallback</span>'

st.markdown(
    '<div class="ps-header">'
    '<div class="ps-brand">'
    '<div class="ps-logo">🗺️</div>'
    "<div>"
    '<div class="ps-title">PathSnap</div>'
    '<div class="ps-sub">Map-matching &amp; route reconstruction · Group 8</div>'
    "</div>"
    "</div>"
    f'<div class="ps-status">{status_html}{fallback_html}{meta_html}</div>'
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Trip selection
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="ps-step"><span class="ps-step-num">1</span>'
    "Select an evaluation trip and decoder</div>",
    unsafe_allow_html=True,
)

if partition and partition != "evaluator_test":
    st.markdown(
        '<div class="ps-callout"><b>Student-test fallback.</b> The ten reserved '
        "evaluator trips are not mounted on Laptop 1, so these are the ten "
        "development trips. Metrics are still computed server-side.</div>",
        unsafe_allow_html=True,
    )

payload, trips_error = fetch_trips(api_url)
trips = (payload or {}).get("trips", [])
partition = (payload or {}).get("partition", partition)

if trips_error:
    st.error(f"Could not load trips: {trips_error}")
    st.stop()
if not trips:
    st.info("No trips found. Place the evaluator trips under `data/evaluator_test/` on Laptop 1.")
    st.stop()

options = {
    f"{t['trip_id']}  ·  {t['n_points']:,} points  ·  {t['n_nodes']:,} nodes": t["trip_id"]
    for t in trips
}

col_trip, col_mode = st.columns([1.65, 1])
with col_trip:
    choice = st.selectbox("Trip", list(options.keys()), label_visibility="collapsed")
    trip_id = options[choice]
with col_mode:
    mode = st.radio(
        "Decoder",
        options=["hmm", "classifier"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

st.caption(
    "**HMM** is the default and enforces network connectivity. "
    "**Classifier-only** is the report comparison; both are scored against the same reference."
)

if st.button("Run prediction", type="primary", use_container_width=True):
    with st.spinner("Candidates → features → classifier → Viterbi → server-side scoring …"):
        try:
            st.session_state["result"] = run_prediction(api_url, trip_id, mode)
            st.session_state["result_key"] = (trip_id, mode)
            st.session_state.pop("predict_error", None)
        except requests.exceptions.Timeout:
            st.session_state["predict_error"] = (
                f"Prediction timed out after {REQUEST_TIMEOUT:.0f}s. Check the API terminal."
            )
        except Exception as exc:  # noqa: BLE001
            st.session_state["predict_error"] = str(exc)

if st.session_state.get("predict_error"):
    st.error(st.session_state["predict_error"])
    if st.button("Retry prediction"):
        st.session_state.pop("predict_error", None)
        with st.spinner("Retrying …"):
            try:
                st.session_state["result"] = run_prediction(api_url, trip_id, mode)
                st.session_state["result_key"] = (trip_id, mode)
            except Exception as exc:  # noqa: BLE001
                st.session_state["predict_error"] = str(exc)
        st.rerun()

result = st.session_state.get("result")
result_key = st.session_state.get("result_key")
if not result:
    st.markdown(
        '<div class="ps-callout-muted">Pick a trip and press <b>Run prediction</b> '
        "to run the full pipeline on Laptop 1.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

if result_key != (trip_id, mode):
    st.caption(
        f"Showing the previous result for **{result['trip_id']}** in "
        f"**{result['mode']}** mode. Press Run prediction to refresh."
    )

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="ps-step"><span class="ps-step-num">2</span>'
    "Server-side evaluation</div>",
    unsafe_allow_html=True,
)
metrics = result.get("metrics", {})

st.markdown(
    '<div class="ps-card-grid">'
    + metric_card("Observation Top-1", fmt_pct(metrics.get("point_top1")), "correct arc per point")
    + metric_card("Route continuity", fmt_pct(metrics.get("strict_continuity")), "consecutive arcs connected")
    + metric_card("Sequence LCS F1", fmt_pct(metrics.get("lcs_f1")), "order-aware route match")
    + metric_card("Arc-set F1", fmt_pct(metrics.get("arc_f1")), "arcs found vs reference")
    + metric_card("Candidate miss", fmt_pct(metrics.get("candidate_miss_rate")), "truth absent from candidates")
    + "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="ps-chips">'
    f'<span class="ps-chip">trip {result["trip_id"]}</span>'
    f'<span class="ps-chip">decoder {result["mode"]}</span>'
    f'<span class="ps-chip">observations {result["n_observations"]:,}</span>'
    f'<span class="ps-chip">matched {result["n_matched"]:,}</span>'
    f'<span class="ps-chip">arcs {result["route_arc_count"]:,}</span>'
    f'<span class="ps-chip">model {result["model_version"]}</span>'
    "</div>",
    unsafe_allow_html=True,
)
st.caption("Every value above is computed on Laptop 1 after prediction; the client receives metrics only.")

with st.expander("Stage timings and full metrics"):
    timings = result.get("timings_ms", {})
    st.write("**Stage timings (ms)**")
    st.table({k: round(float(v), 1) for k, v in timings.items()})
    st.write("**All metrics**")
    st.json(metrics)

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="ps-step"><span class="ps-step-num">3</span>Predicted route</div>',
    unsafe_allow_html=True,
)
predicted = result.get("predicted_route", [])
observed = result.get("observed_points", [])

if predicted:
    lon = [p[0] for p in predicted]
    lat = [p[1] for p in predicted]
    fmap = folium.Map(
        location=[sum(lat) / len(lat), sum(lon) / len(lon)],
        zoom_start=13,
        control_scale=True,
        tiles=None,
    )
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
        ),
        attr='Tiles &copy; <a href="https://www.esri.com/">Esri</a>',
        name="Dark basemap",
        control=False,
    ).add_to(fmap)
    # Dark-theme the Leaflet controls so the map does not flash a white panel.
    fmap.get_root().header.add_child(
        folium.Element(
            "<style>"
            ".leaflet-control-layers,.leaflet-bar a{"
            "background:#151b23!important;color:#e6edf3!important;"
            "border-color:#263043!important;}"
            ".leaflet-control-layers label{color:#e6edf3;}"
            ".leaflet-control-attribution{background:rgba(13,17,23,.75)!important;"
            "color:#93a1b5!important;}"
            ".leaflet-control-attribution a{color:#7c9cff!important;}"
            "</style>"
        )
    )
    observed_layer = folium.FeatureGroup(name="GPS observations", show=True)
    for lon_i, lat_i in observed:
        folium.CircleMarker(
            location=[lat_i, lon_i],
            radius=2.6,
            color="#a5d6ff",
            fill=True,
            fill_color="#a5d6ff",
            fill_opacity=0.9,
            weight=0,
        ).add_to(observed_layer)
    observed_layer.add_to(fmap)

    pred_layer = folium.FeatureGroup(name="Predicted route", show=True)
    folium.PolyLine(
        locations=[[p[1], p[0]] for p in predicted],
        color="#ff7b72",
        weight=5,
        opacity=0.95,
        tooltip="Predicted route",
    ).add_to(pred_layer)
    pred_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[lat_i, lon_i] for lon_i, lat_i in predicted], padding=(24, 24))

    st.markdown(
        '<div class="ps-legend">'
        '<span class="ps-dot-blue">●</span> GPS observations &nbsp;&nbsp;'
        '<span class="ps-dot-red">━</span> Predicted route'
        "</div>",
        unsafe_allow_html=True,
    )
    st.iframe(fmap._repr_html_(), height=720)
    st.download_button(
        "Download predicted route (GeoJSON)",
        data=json.dumps(
            {
                "type": "Feature",
                "properties": {"trip_id": result["trip_id"], "decoder": result["mode"]},
                "geometry": {"type": "LineString", "coordinates": predicted},
            }
        ),
        file_name=f"pathsnap_{result['trip_id']}_{result['mode']}.geojson",
        mime="application/geo+json",
    )
else:
    st.warning("The API returned no predicted route.")

with st.expander("Raw API response"):
    st.json(result)
