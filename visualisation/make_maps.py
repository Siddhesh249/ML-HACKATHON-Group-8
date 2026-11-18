#!/usr/bin/env python3
"""Generate Folium maps for the report and the first viva (T21; T28 extends it).

Phase 1 map: GPS observations + trip-local road network + reference route,
written to ``results/maps/<partition>_<trip>_reference.html`` with a PNG
screenshot in ``results/figures/`` (headless Chrome, if available).

These maps show the reference route, so they are **report/EDA maps only**.
The live evaluation UI must never display the reference route.

Phase 2 (T28): pass ``--artifact-dir`` to add the classifier/HMM prediction
layer from the exported model.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import config, io, visualize  # noqa: E402

DEFAULT_TRIPS = ["train/00000046", "train/00000033"]
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
    "chromium-browser",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(config.DATA_ROOT))
    parser.add_argument("--trips", nargs="+", default=DEFAULT_TRIPS, help="partition/trip_id")
    parser.add_argument("--maps-dir", default=str(config.RESULTS_DIR / "maps"))
    parser.add_argument("--figures-dir", default=str(config.RESULTS_DIR / "figures"))
    parser.add_argument("--artifact-dir", default=None, help="Phase 2: add the predicted route from this artifact")
    parser.add_argument("--mode", choices=["hmm", "classifier"], default="hmm")
    parser.add_argument("--no-screenshot", action="store_true")
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    maps_dir, figures_dir = Path(args.maps_dir), Path(args.figures_dir)
    chrome = None if args.no_screenshot else find_chrome()
    if not args.no_screenshot and chrome is None:
        print("No Chrome/Chromium found: writing HTML only; take the screenshot manually.")

    for key in args.trips:
        partition, trip_id = key.split("/")
        if partition == config.EVALUATION_PARTITION:
            raise SystemExit("Refusing to draw a reference-route map for evaluator trips.")
        trip = io.load_trip(Path(args.data_root) / partition / trip_id, with_route=True)
        title = (f"PathSnap · {partition}/{trip_id} · {len(trip.track)} GPS points · "
                 f"{trip.network.n_arcs} arcs · reference route {trip.route.size} arcs")
        fmap = visualize.trip_map(trip, reference_arcs=trip.route, show_reference=True, title=title)
        html = visualize.save_map(fmap, maps_dir / f"{partition}_{trip_id}_reference.html")
        size_mb = html.stat().st_size / 1e6
        print(f"{key}: {html.relative_to(ROOT)} ({size_mb:.1f} MB)")
        if chrome:
            png = figures_dir / f"map_{partition}_{trip_id}_reference.png"
            if screenshot(html, png, chrome):
                print(f"  screenshot: {png.relative_to(ROOT)}")
        if args.artifact_dir:
            raise SystemExit("Prediction overlay is Phase 2 (T28); not implemented in the Phase-1 map.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
