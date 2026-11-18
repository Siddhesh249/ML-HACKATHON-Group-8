"""Record warm transition latency for one real trip."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

from pathsnap import candidates, config, io, hmm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT)
    parser.add_argument("--partition", default="train")
    parser.add_argument("--trip-id", required=True)
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "hmm" / "transition_benchmark.csv")
    args = parser.parse_args()

    trip = io.load_trip(args.data_root / args.partition / args.trip_id, with_route=False)
    cand = candidates.CandidateGenerator(trip.network).generate(trip.track)
    context = hmm.DecoderContext(trip, cand, hmm.HMMConfig())
    valid = [i for i in range(len(cand)) if cand.offsets[i + 1] > cand.offsets[i]]
    pairs = list(zip(valid, valid[1:]))
    start = time.perf_counter()
    for prev, cur in pairs:
        context.transition(prev, cur)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "commit": commit,
        "partition": args.partition,
        "trip_id": args.trip_id,
        "n_observations": len(cand),
        "n_candidates": len(cand.arc_ids),
        "corridor_nodes": context.graph.shape[0],
        "cached_dijkstra_sources": len(context._dist_cache),
        "transition_pairs": len(pairs),
        "warm_transition_ms": elapsed_ms,
    }
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
