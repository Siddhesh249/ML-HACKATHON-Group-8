#!/usr/bin/env python3
"""T25: tune HMM parameters on train or student-test only."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import config, hmm, io  # noqa: E402
from hmm_eval_common import git_commit, prepare_trip, evaluate_prepared  # noqa: E402


GRID = {
    "lam": (0.0, 0.5, 1.0),
    "sigma_m": (8.0, 15.0),
    "beta_m": (10.0, 30.0),
    "max_speed_mps": (55.0,),
    "gap_reset_s": (30.0, 60.0),
    "uturn_penalty": (0.0, 2.0),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(config.DATA_ROOT))
    parser.add_argument("--artifact-dir", default=str(config.ARTIFACT_DIR))
    parser.add_argument("--partition", choices=("train", "student_test"), default="student_test")
    parser.add_argument("--out", default=str(config.RESULTS_DIR / "hmm" / "decoder_tuning.csv"))
    parser.add_argument("--config-out", default=str(config.ARTIFACT_DIR / "hmm_config.json"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    metas = io.discover_trips(args.data_root, partition=args.partition)
    if args.limit:
        metas = metas[: args.limit]
    if not metas:
        raise RuntimeError(f"No trips found in permitted partition {args.partition!r} under {args.data_root}")

    prepared = [prepare_trip(args.data_root, args.artifact_dir, meta.trip_id) for meta in metas]
    rows: list[dict] = []
    keys = tuple(GRID)
    for values in itertools.product(*(GRID[key] for key in keys)):
        params = dict(zip(keys, values))
        decoder_config = hmm.HMMConfig(**params)
        trip_rows = [evaluate_prepared(item, decoder_config, "hmm") for item in prepared]
        means = {
            name: float(np.nanmean([row.get(name, np.nan) for row in trip_rows]))
            for name in ("point_top1", "strict_continuity", "relaxed_continuity", "lcs_f1", "arc_f1", "decode_ms")
        }
        rows.append({
            "commit": git_commit(),
            "partition": args.partition,
            "n_trips": len(prepared),
            **params,
            **means,
            "objective": means["point_top1"] + 0.25 * means["strict_continuity"] + 0.25 * means["lcs_f1"],
        })

    rows.sort(key=lambda row: row["objective"], reverse=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    best = {key: rows[0][key] for key in keys}
    config_out = Path(args.config_out)
    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} configurations to {out}; best objective={rows[0]['objective']:.6f}")
    print(f"Wrote tuned decoder config to {config_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
