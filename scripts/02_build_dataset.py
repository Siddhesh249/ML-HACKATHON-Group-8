#!/usr/bin/env python3
"""Stage 2 — build the cached candidate/feature dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import config, dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(config.DATA_ROOT))
    parser.add_argument("--partitions", nargs="+", default=["train"])
    parser.add_argument("--out", default=str(config.CACHE_DIR / "dataset.npz"))
    parser.add_argument("--cache-dir", default=str(config.CACHE_DIR))
    parser.add_argument("--limit", type=int, default=None, help="Build only the first N trips (debug)")
    parser.add_argument("--no-graph", action="store_true", help="Disable graph-distance features")
    parser.add_argument("--force", action="store_true", help="Rebuild per-trip caches")
    parser.add_argument("--folds", type=int, default=5, help="GroupKFold splits recorded in the summary")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Do not rebuild; write results/models/dataset_summary.json + cv_splits.csv from the cache",
    )
    return parser.parse_args()


def write_summary(args: argparse.Namespace) -> None:
    command = " ".join(["python3", "scripts/02_build_dataset.py", *sys.argv[1:]])
    summary = dataset.write_dataset_summary(
        args.out, config.RESULTS_DIR / "models", n_splits=args.folds, command=command
    )
    print(json.dumps({k: summary[k] for k in ("n_trips", "n_rows", "n_positive", "checks")}, indent=2))
    print(f"Dataset summary written to {config.RESULTS_DIR / 'models'}")


def main() -> int:
    args = parse_args()
    if args.summary_only:
        write_summary(args)
        return 0
    meta = dataset.build_dataset(
        data_root=args.data_root,
        partitions=tuple(args.partitions),
        out_path=args.out,
        cache_dir=args.cache_dir,
        limit=args.limit,
        use_graph=not args.no_graph,
        force=args.force,
    )
    print(json.dumps({k: v for k, v in meta.items() if k != "stats"}, indent=2))
    print(f"Dataset written to {args.out}")
    dataset.write_stat_reports(
        meta,
        labels_dir=config.RESULTS_DIR / "labels",
        eda_dir=config.RESULTS_DIR / "eda",
    )
    print(f"Label reports written to {config.RESULTS_DIR / 'labels'}")
    if tuple(args.partitions) == ("train",):
        write_summary(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
