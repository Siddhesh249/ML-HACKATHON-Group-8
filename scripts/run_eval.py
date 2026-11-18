#!/usr/bin/env python3
"""Batch evaluation through the real Predictor: classifier-only vs HMM.

This is the integration runner used by D for T32 and by C for the decoder
comparison. It never bypasses the API code path: it calls the same Predictor
that FastAPI uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import config, io  # noqa: E402
from backend.service import Predictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(config.DATA_ROOT))
    parser.add_argument("--artifact-dir", default=str(config.ARTIFACT_DIR))
    parser.add_argument("--partition", default="student_test")
    parser.add_argument("--modes", nargs="+", default=["hmm", "classifier"])
    parser.add_argument("--out", default=str(config.RESULTS_DIR / "integration" / "run_log.csv"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def run_evaluation(
    data_root: str,
    artifact_dir: str,
    partition: str,
    modes: list[str],
    out_path: str,
    limit: int | None = None,
) -> pd.DataFrame:
    predictor = Predictor(data_root=data_root, artifact_dir=artifact_dir).load()
    if not predictor.ready:
        raise RuntimeError(predictor.load_error or "Model is not ready")

    metas = io.discover_trips(data_root, partition=partition)
    if limit:
        metas = metas[:limit]
    if not metas:
        raise RuntimeError(f"No trips in partition {partition} under {data_root}")

    rows = []
    for meta in metas:
        for mode in modes:
            print(f"== {partition}/{meta.trip_id} mode={mode}", flush=True)
            result = predictor.predict(meta.trip_id, mode=mode)
            row = {
                "trip_id": meta.trip_id,
                "partition": partition,
                "mode": mode,
                "model_name": result["model_name"],
                "model_version": result["model_version"],
                "n_observations": result["n_observations"],
                "n_matched": result["n_matched"],
                "route_arc_count": result["route_arc_count"],
                "total_ms": result["timings_ms"].get("total_ms"),
                **{f"metric_{k}": v for k, v in result["metrics"].items()},
            }
            rows.append(row)
            print(
                f"   top1={result['metrics'].get('point_top1')} "
                f"continuity={result['metrics'].get('strict_continuity')} "
                f"lcs_f1={result['metrics'].get('lcs_f1')} "
                f"ms={result['timings_ms'].get('total_ms'):.0f}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump({"n_trips": len(metas), "modes": modes, "rows": len(rows)}, fh, indent=2)
    return frame


def main() -> int:
    args = parse_args()
    frame = run_evaluation(
        data_root=args.data_root,
        artifact_dir=args.artifact_dir,
        partition=args.partition,
        modes=args.modes,
        out_path=args.out,
        limit=args.limit,
    )
    if frame.empty:
        print("No results")
        return 1
    keep = [
        c
        for c in frame.columns
        if c.startswith("metric_") or c in {"trip_id", "mode", "total_ms"}
    ]
    print("\n" + frame[keep].to_string(index=False))
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
