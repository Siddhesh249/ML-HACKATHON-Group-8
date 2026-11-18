#!/usr/bin/env python3
"""Generate Folium maps for the report and the viva (T21, T28).

Two kinds of map:

* ``--trips``   (T21) GPS + trip-local network + reference route.
* ``--predict`` (T28) GPS + network + reference + **classifier-only** and
  **HMM** predicted routes as separate layers, scored server-side. Writes the
  interactive HTML, a full-trip screenshot and a zoomed screenshot of the
  window where the two decoders disagree most, plus a metrics CSV.

The prediction runs exactly like the API (``backend/service.py``): candidates
→ features (no labels) → exported classifier → ``hmm.decode``. The reference
route is loaded only afterwards, to draw it and to score. These maps show the
reference route, so they are report/viva maps only — the live evaluation UI
must never display it. Evaluator trips are refused.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import candidates as cand_mod, config, features, hmm, io, models, scoring, visualize  # noqa: E402

DEFAULT_TRIPS = ["train/00000046", "train/00000033"]
DEFAULT_PREDICT = ["student_test/00000052", "student_test/00000056", "student_test/00000077"]
CLASSIFIER_COLOUR = "#ef6c00"  # orange
HMM_COLOUR = "#8e24aa"         # purple
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
    "chromium-browser",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(config.DATA_ROOT))
    parser.add_argument("--trips", nargs="*", default=None, help="T21 reference maps: partition/trip_id")
    parser.add_argument("--predict", nargs="*", default=None, help="T28 prediction maps: partition/trip_id")
    parser.add_argument("--maps-dir", default=str(config.RESULTS_DIR / "maps"))
    parser.add_argument("--figures-dir", default=str(config.RESULTS_DIR / "figures"))
    parser.add_argument("--artifact-dir", default=str(config.ARTIFACT_DIR))
    parser.add_argument("--zoom-window", type=int, default=90, help="observations in the zoomed window")
    parser.add_argument("--no-screenshot", action="store_true")
    args = parser.parse_args()
    if args.trips is None and args.predict is None:
        args.trips, args.predict = DEFAULT_TRIPS, DEFAULT_PREDICT
    # A bare flag (``--predict`` with no IDs) means "the defaults".
    if args.trips == []:
        args.trips = DEFAULT_TRIPS
    if args.predict == []:
        args.predict = DEFAULT_PREDICT
    return args


def find_chrome() -> str | None:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists() or shutil.which(candidate):
            return candidate
    return None


def screenshot(html: Path, png: Path, chrome: str) -> bool:
    png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--window-size=1400,900", "--virtual-time-budget=15000",
        f"--screenshot={png}", html.resolve().as_uri(),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except Exception as exc:  # noqa: BLE001 - screenshots are best-effort
        print(f"  screenshot failed: {exc}")
        return False
    return png.exists()


def guard(partition: str) -> None:
    if partition == config.EVALUATION_PARTITION:
        raise SystemExit("Refusing to draw a reference-route map for evaluator trips.")


def reference_map(key: str, args, chrome: str | None) -> None:
    partition, trip_id = key.split("/")
    guard(partition)
    trip = io.load_trip(Path(args.data_root) / partition / trip_id, with_route=True)
    title = (f"PathSnap · {partition}/{trip_id} · {len(trip.track)} GPS points · "
             f"{trip.network.n_arcs} arcs · reference route {trip.route.size} arcs")
    fmap = visualize.trip_map(trip, reference_arcs=trip.route, show_reference=True, title=title)
    html = visualize.save_map(fmap, Path(args.maps_dir) / f"{partition}_{trip_id}_reference.html")
    print(f"{key}: {html.relative_to(ROOT)} ({html.stat().st_size / 1e6:.1f} MB)")
    if chrome:
        png = Path(args.figures_dir) / f"map_{partition}_{trip_id}_reference.png"
        if screenshot(html, png, chrome):
            print(f"  screenshot: {png.relative_to(ROOT)}")


class Pipeline:
    """The API's prediction path, with arc IDs exposed for drawing."""

    def __init__(self, artifact_dir: str):
        self.artifact = models.load_artifact(Path(artifact_dir) / "model.joblib")
        if tuple(self.artifact["feature_names"]) != tuple(config.FEATURE_NAMES):
            raise SystemExit("Artifact feature contract differs from config.FEATURE_NAMES")
        cfg_path = Path(artifact_dir) / "hmm_config.json"
        self.hmm_cfg = hmm.HMMConfig(**json.loads(cfg_path.read_text())) if cfg_path.exists() else hmm.HMMConfig()
        self.cfg = self.artifact.get("config", {})

    def run(self, trip_dir: Path):
        trip = io.load_trip(trip_dir, with_route=False)  # prediction never sees the route
        gen = cand_mod.CandidateGenerator(
            trip.network,
            radius_m=float(self.cfg.get("candidate_radius_m", config.CANDIDATE_RADIUS_M)),
            top_k=int(self.cfg.get("candidate_top_k", config.CANDIDATE_TOP_K)),
        )
        cands = gen.generate(trip.track)
        fm = features.build_features(
            trip, cands, alignment=None,
            options=features.FeatureOptions(use_graph=bool(self.cfg.get("use_graph_features", True))),
        )
        probs = np.asarray(self.artifact["model"].predict_proba(fm.X)[:, 1], dtype=float)
        out = {mode: hmm.decode(trip, cands, probs, mode=mode, cfg=self.hmm_cfg).arc_ids
               for mode in ("classifier", "hmm")}
        return trip, cands, out


def disagreement_bounds(trip, a: np.ndarray, b: np.ndarray, window: int):
    diff = (a != b).astype(int)
    if diff.sum() == 0 or len(diff) < 2:
        return None, 0
    window = min(window, len(diff))
    rolling = np.convolve(diff, np.ones(window, dtype=int), mode="valid")
    start = int(np.argmax(rolling))
    sl = slice(start, start + window)
    lat, lon = trip.track.lonlat[sl, 1], trip.track.lonlat[sl, 0]
    pad = 0.0012
    return (float(lat.min() - pad), float(lon.min() - pad), float(lat.max() + pad), float(lon.max() + pad)), \
        int(rolling[start])


def prediction_map(key: str, args, chrome: str | None, pipeline: Pipeline) -> dict:
    partition, trip_id = key.split("/")
    guard(partition)
    trip_dir = Path(args.data_root) / partition / trip_id
    trip, cands, preds = pipeline.run(trip_dir)

    # ---- scoring: first point at which the reference route is read --------
    trip_ref, alignment = scoring.load_alignment(trip_dir)
    scores = {mode: scoring.evaluate_predictions(trip_ref, cands, alignment, arcs).metrics
              for mode, arcs in preds.items()}
    c, h = scores["classifier"], scores["hmm"]
    title = (f"PathSnap · {partition}/{trip_id} · point accuracy: classifier {c['point_top1']:.3f}, "
             f"HMM {h['point_top1']:.3f} · strict continuity: {c['strict_continuity']:.2f} → "
             f"{h['strict_continuity']:.2f} · HMM config: artifacts/hmm_config.json")
    layers = {
        "Classifier-only route": (preds["classifier"], CLASSIFIER_COLOUR),
        "HMM route": (preds["hmm"], HMM_COLOUR),
    }
    stem = f"{partition}_{trip_id}_classifier_vs_hmm"
    html = visualize.save_map(
        visualize.trip_map(trip_ref, reference_arcs=trip_ref.route, show_reference=True,
                           predictions=layers, title=title, max_observed=1500),
        Path(args.maps_dir) / f"{stem}.html",
    )
    print(f"{key}: {html.relative_to(ROOT)} ({html.stat().st_size / 1e6:.1f} MB)  "
          f"classifier {c['point_top1']:.3f} / HMM {h['point_top1']:.3f}")
    bounds, n_diff = disagreement_bounds(trip_ref, preds["classifier"], preds["hmm"], args.zoom_window)
    zoom_html = None
    if bounds is not None and chrome:
        # The zoom view only exists to be screenshotted; keep its HTML out of results/.
        import tempfile

        zoom_html = visualize.save_map(
            visualize.trip_map(trip_ref, reference_arcs=trip_ref.route, show_reference=True,
                               predictions=layers, bounds=bounds, max_observed=4000,
                               title=title + f" · zoom: {n_diff}/{args.zoom_window} observations disagree"),
            Path(tempfile.mkdtemp(prefix="pathsnap_zoom_")) / f"{stem}_zoom.html",
        )
    if chrome:
        for src, suffix in ((html, ""), (zoom_html, "_zoom")):
            if src is None:
                continue
            png = Path(args.figures_dir) / f"map_{stem}{suffix}.png"
            if screenshot(src, png, chrome):
                print(f"  screenshot: {png.relative_to(ROOT)}")
    row = {"partition": partition, "trip_id": trip_id, "n_observations": len(trip.track),
           "obs_decoders_disagree": int((preds["classifier"] != preds["hmm"]).sum()),
           "zoom_window_disagree": n_diff}
    for mode, m in scores.items():
        for k in ("point_top1", "strict_continuity", "relaxed_continuity", "lcs_f1", "arc_f1"):
            row[f"{k}_{mode}"] = m.get(k)
    # Failure analysis: points the HMM gets wrong that the classifier got right,
    # and how many of those are the opposite-direction twin of the correct arc
    # (same road, reversed direction — identical on the map).
    c_arcs, h_arcs, arcs = preds["classifier"], preds["hmm"], trip_ref.network.arcs
    labelled = alignment.confidence & (alignment.arc_ids >= 0)
    lost = np.flatnonzero(labelled & (h_arcs != alignment.arc_ids) & (c_arcs == alignment.arc_ids) & (h_arcs >= 0))
    twin = (arcs[c_arcs[lost], 0] == arcs[h_arcs[lost], 1]) & (arcs[c_arcs[lost], 1] == arcs[h_arcs[lost], 0])
    row["hmm_lost_points"] = int(lost.size)
    row["hmm_lost_opposite_direction_share"] = float(twin.mean()) if lost.size else 0.0
    row["html"] = str(html.relative_to(ROOT))
    return row


def main() -> int:
    args = parse_args()
    chrome = None if args.no_screenshot else find_chrome()
    if not args.no_screenshot and chrome is None:
        print("No Chrome/Chromium found: writing HTML only; take the screenshots manually.")
    for key in args.trips or []:
        reference_map(key, args, chrome)
    if args.predict:
        pipeline = Pipeline(args.artifact_dir)
        rows = [prediction_map(key, args, chrome, pipeline) for key in args.predict]
        out = Path(args.maps_dir) / "prediction_maps.csv"
        frame = pd.DataFrame(rows)
        frame.insert(0, "model_version", pipeline.artifact.get("model_version"))
        frame.insert(1, "hmm_config", json.dumps(pipeline.hmm_cfg.to_dict()))
        frame.to_csv(out, index=False)
        print(f"metrics: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
