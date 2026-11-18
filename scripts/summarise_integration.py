#!/usr/bin/env python3
"""Rebuild the integration summaries from ``results/integration/run_log.csv``.

``scripts/run_eval.py`` writes the per-request run log; this script derives the
files quoted in the viva/report from it, so every number is regenerable:
``summary.json``, ``per_trip_comparison.csv`` and ``stage_timings.csv``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "integration"
STAGES = ["ms_load_ms", "ms_candidates_ms", "ms_features_ms", "ms_model_ms", "ms_decode_ms", "ms_scoring_ms", "ms_total_ms"]
METRICS = ["metric_point_top1", "metric_strict_continuity", "metric_lcs_f1"]


def main() -> int:
    log = pd.read_csv(OUT / "run_log.csv", dtype={"trip_id": str})
    log["trip_id"] = log["trip_id"].str.zfill(8)

    timings = []
    for mode, g in log.groupby("mode"):
        row = {"mode": mode}
        for s in STAGES:
            row[f"{s}_p50"] = float(g[s].quantile(0.5))
            row[f"{s}_p95"] = float(g[s].quantile(0.95))
            row[f"{s}_max"] = float(g[s].max())
        timings.append(row)
    pd.DataFrame(timings).to_csv(OUT / "stage_timings.csv", index=False)

    wide = log.pivot(index="trip_id", columns="mode", values=METRICS + ["ms_total_ms"])
    wide.columns = [f"{m}_{mode}" for m, mode in wide.columns]
    cols = [f"{m}_{mode}" for m in METRICS + ["ms_total_ms"] for mode in ("classifier", "hmm")]
    per_trip = wide[cols].reset_index()
    per_trip["cont_improvement"] = per_trip["metric_strict_continuity_hmm"] - per_trip["metric_strict_continuity_classifier"]
    per_trip["top1_delta"] = per_trip["metric_point_top1_hmm"] - per_trip["metric_point_top1_classifier"]
    per_trip.to_csv(OUT / "per_trip_comparison.csv", index=False)

    aggregate = []
    for mode, g in log.groupby("mode"):
        aggregate.append({
            "mode": mode, "n_trips": float(len(g)),
            "top1_mean": float(g["metric_point_top1"].mean()), "top1_median": float(g["metric_point_top1"].median()),
            "continuity_mean": float(g["metric_strict_continuity"].mean()),
            "lcs_f1_mean": float(g["metric_lcs_f1"].mean()), "arc_f1_mean": float(g["metric_arc_f1"].mean()),
        })
    worst = log.loc[log["ms_total_ms"].idxmax()]
    summary = {
        "n_trips": int(log["trip_id"].nunique()), "n_rows": int(len(log)), "modes": sorted(log["mode"].unique()),
        "all_trips_passed": bool(len(log) == log["trip_id"].nunique() * log["mode"].nunique()),
        "aggregate": aggregate, "timings": timings,
        "worst_latency_ms": float(worst["ms_total_ms"]), "worst_latency_trip": f"{worst['trip_id']} ({worst['mode']})",
        "hmm_continuity_better_count": int((per_trip["cont_improvement"] > 0).sum()),
        "hmm_top1_worse_count": int((per_trip["top1_delta"] < 0).sum()),
        "source": "results/integration/run_log.csv (scripts/run_eval.py) via scripts/summarise_integration.py",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("all_trips_passed", "aggregate", "worst_latency_ms", "worst_latency_trip",
                                              "hmm_continuity_better_count", "hmm_top1_worse_count")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
