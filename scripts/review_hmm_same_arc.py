#!/usr/bin/env python3
"""B review of T17 — evidence for the same-arc backward-jitter issue.

``pathsnap.hmm.DecoderContext.transition`` treats any backward projection
progress on the *same* arc as infeasible. This script measures how often the
reference says the vehicle stayed on one arc between two consecutive
observations (both confidently labelled, no gap reset, truth in both candidate
sets) while the projection of the raw GPS point moved backwards along that arc
— i.e. how often the decoder forbids the true transition.

Train trips only; reads the reference route for measurement, never for
prediction. Writes ``results/models/hmm_review_same_arc.csv`` (per trip) and
``hmm_review_same_arc.json`` (pooled).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import candidates as cand_mod, config, io, labels, network as net  # noqa: E402

TOLERANCES_M = (0.0, 2.0, 5.0, 10.0, 15.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(config.DATA_ROOT))
    parser.add_argument("--partition", default="train")
    parser.add_argument("--out-dir", default=str(config.RESULTS_DIR / "models"))
    return parser.parse_args()


def trip_backsteps(trip) -> np.ndarray:
    """Backward distance (m, >= 0) for every true same-arc consecutive pair."""
    alignment = labels.align_track_to_route(trip)
    cands = cand_mod.CandidateGenerator(trip.network).generate(trip.track)
    _, _, lengths, _, _ = net.arc_geometry(trip.network, origin=cands.origin)
    lab, conf, t = alignment.arc_ids, alignment.confidence, trip.track.t
    out = []
    for i in range(1, len(trip.track)):
        a = lab[i]
        if a < 0 or a != lab[i - 1] or not (conf[i] and conf[i - 1]):
            continue
        if t[i] - t[i - 1] >= config.HMM_GAP_RESET_S:
            continue  # gap reset: the same-arc rule is not applied
        lo0, hi0 = cands.offsets[i - 1], cands.offsets[i]
        lo1, hi1 = cands.offsets[i], cands.offsets[i + 1]
        h0 = np.flatnonzero(cands.arc_ids[lo0:hi0] == a)
        h1 = np.flatnonzero(cands.arc_ids[lo1:hi1] == a)
        if not h0.size or not h1.size:
            continue
        r0 = np.clip(cands.projection_ratio[lo0 + h0[0]], 0.0, 1.0)
        r1 = np.clip(cands.projection_ratio[lo1 + h1[0]], 0.0, 1.0)
        out.append(max(0.0, float((r0 - r1) * lengths[a])))
    return np.asarray(out)


def main() -> int:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, pooled = [], []
    for meta in io.discover_trips(args.data_root, partition=args.partition):
        back = trip_backsteps(io.load_trip(meta.directory, with_route=True))
        pooled.append(back)
        row = {"trip_id": meta.trip_id, "partition": meta.partition, "same_arc_pairs": int(back.size)}
        for tol in TOLERANCES_M:
            key = "frac_infeasible_now" if tol == 0 else f"frac_backward_gt_{int(tol)}m"
            row[key] = float(np.mean(back > tol)) if back.size else float("nan")
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("frac_infeasible_now", ascending=False)
    frame.to_csv(out / "hmm_review_same_arc.csv", index=False)

    allb = np.concatenate(pooled) if pooled else np.zeros(0)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(["python3", "scripts/review_hmm_same_arc.py", *sys.argv[1:]]),
        "partition": args.partition,
        "n_trips": int(len(frame)),
        "same_arc_pairs": int(allb.size),
        "frac_infeasible_now": float(np.mean(allb > 0.0)),
        **{f"frac_backward_gt_{int(t)}m": float(np.mean(allb > t)) for t in TOLERANCES_M if t > 0},
        "backward_m_p50_given_backward": float(np.median(allb[allb > 0])) if (allb > 0).any() else 0.0,
        "backward_m_p95_given_backward": float(np.percentile(allb[allb > 0], 95)) if (allb > 0).any() else 0.0,
        "worst_trips": frame.head(5)[["trip_id", "frac_infeasible_now"]].to_dict("records"),
    }
    with open(out / "hmm_review_same_arc.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
