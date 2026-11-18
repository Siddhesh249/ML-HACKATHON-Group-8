#!/usr/bin/env python3
"""T26: compare classifier-only and tuned HMM on permitted trips."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import config, io  # noqa: E402
from hmm_eval_common import evaluate_prepared, load_hmm_config, prepare_trip  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(config.DATA_ROOT))
    parser.add_argument("--artifact-dir", default=str(config.ARTIFACT_DIR))
    parser.add_argument("--partition", choices=("train", "student_test"), default="student_test")
    parser.add_argument("--out", default=str(config.RESULTS_DIR / "hmm" / "classifier_vs_hmm.csv"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    metas = io.discover_trips(args.data_root, partition=args.partition)
    if args.limit:
        metas = metas[: args.limit]
    if not metas:
        raise RuntimeError(f"No trips found in permitted partition {args.partition!r} under {args.data_root}")

    decoder_config = load_hmm_config(Path(args.artifact_dir) / "hmm_config.json")
    rows = []
    for meta in metas:
        prepared = prepare_trip(args.data_root, args.artifact_dir, meta.trip_id)
        for mode in ("classifier", "hmm"):
            rows.append(evaluate_prepared(prepared, decoder_config, mode))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
