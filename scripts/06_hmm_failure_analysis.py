#!/usr/bin/env python3
"""T27: select decoder failures and generate report maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import config, hmm  # noqa: E402
from pathsnap.contracts import DataError  # noqa: E402
from hmm_eval_common import load_hmm_config, prepare_trip  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(config.DATA_ROOT))
    parser.add_argument("--artifact-dir", default=str(config.ARTIFACT_DIR))
    parser.add_argument("--comparison", default=str(config.RESULTS_DIR / "hmm" / "classifier_vs_hmm.csv"))
    parser.add_argument("--out", default=str(config.RESULTS_DIR / "hmm" / "failures.md"))
    parser.add_argument("--map-dir", default=str(config.RESULTS_DIR / "hmm" / "failure_maps"))
    args = parser.parse_args()
    comparison = pd.read_csv(args.comparison)
    if "mode" not in comparison or comparison["mode"].nunique() < 2:
        raise ValueError("comparison CSV must contain classifier and hmm rows")
    wide = comparison.pivot(index="trip_id", columns="mode")
    wide.columns = [f"{metric}_{mode}" for metric, mode in wide.columns]
    wide = wide.reset_index()
    wide["top1_delta"] = wide["point_top1_hmm"] - wide["point_top1_classifier"]
    wide["continuity_delta"] = wide["strict_continuity_hmm"] - wide["strict_continuity_classifier"]
    wide["lcs_delta"] = wide["lcs_f1_hmm"] - wide["lcs_f1_classifier"]
    selected: set[str] = set()

    def pick(name: str, frame: pd.DataFrame) -> tuple[str, pd.Series]:
        for _, candidate in frame.iterrows():
            trip = f"{int(candidate['trip_id']):08d}"
            if trip not in selected:
                selected.add(trip)
                return name, candidate
        raise RuntimeError(f"Could not select a distinct failure case for {name}")

    cases = dict([
        pick("largest_top1_regression", wide.sort_values("top1_delta")),
        pick("largest_continuity_recovery", wide.sort_values("continuity_delta", ascending=False)),
        pick("largest_sequence_regression", wide.sort_values("lcs_delta")),
    ])
    cfg = load_hmm_config(Path(args.artifact_dir) / "hmm_config.json")
    map_dir = Path(args.map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HMM failure analysis", "",
        "Cases are selected mechanically from the classifier-vs-HMM table.", "",
        "| Case | Trip | Classifier Top-1 | HMM Top-1 | Classifier continuity | HMM continuity | Classifier LCS-F1 | HMM LCS-F1 | Map |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in cases.items():
        trip_id = f"{int(row['trip_id']):08d}"
        map_path = map_dir / f"{trip_id}_{name}.html"
        try:
            prepared = prepare_trip(args.data_root, args.artifact_dir, trip_id)
            prediction = hmm.decode(prepared.trip, prepared.candidates, prepared.probabilities, mode="hmm", cfg=cfg)
            from pathsnap import visualize

            visualize.save_map(
                visualize.trip_map(
                    prepared.scored_trip,
                    predicted_arcs=prediction.arc_ids,
                    reference_arcs=prepared.scored_trip.route,
                    title=f"{trip_id}: {name}",
                    show_reference=True,
                ),
                map_path,
            )
            map_label = str(map_path)
        except (DataError, FileNotFoundError):
            map_label = "not generated: raw trip files unavailable"
        lines.append(
            f"| {name} | {trip_id} | {row['point_top1_classifier']:.4f} | {row['point_top1_hmm']:.4f} | "
            f"{row['strict_continuity_classifier']:.4f} | {row['strict_continuity_hmm']:.4f} | "
            f"{row['lcs_f1_classifier']:.4f} | {row['lcs_f1_hmm']:.4f} | `{map_label}` |"
        )
    lines += [
        "", "## Interpretation checklist", "",
        "- A Top-1 regression means smoothing selected a connected but wrong road; inspect junctions and parallel roads.",
        "- A continuity recovery means the HMM removed an independent-classifier teleport.",
        "- A sequence regression means continuity alone was not enough; inspect gaps, missing candidates, or over-smoothing.",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
