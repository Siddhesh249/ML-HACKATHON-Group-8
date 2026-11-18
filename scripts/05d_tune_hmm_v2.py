#!/usr/bin/env python3
"""HMM decoder retuning (T25 v2): tune on train trips, report on student-test.

Why a second pass: the first grid (``results/hmm/decoder_tuning.csv``) searched
λ, σ, β, gap reset and U-turn penalty on the 10 student-test trips and then
reported on the same trips. Its best HMM still lost Top-1 to classifier-only,
and the dominant error is structural: same-arc backward GPS jitter is
infeasible, so the decoder flips to the opposite-direction twin arc
(`docs/reviews/T17_hmm_review_B.md`, `results/maps/prediction_maps.csv`).

This script

1. adds the same-arc backward tolerance ``same_arc_backward_tol_m`` (τ) to the
   grid and widens β and the U-turn penalty;
2. **selects on train trips** using the classifier's *out-of-fold* probabilities
   (`cache/oof_hist_gradient_boosting.npy`), so neither the classifier nor the
   decoder has seen the selection trips;
3. reports classifier-only vs the previous config vs the new config on the 10
   student-test trips, which played no part in the choice;
4. checks that its fast vectorised decoder reproduces ``pathsnap.hmm.decode_hmm``
   exactly before trusting any number.

Evaluator trips are never read.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import candidates as cand_mod, config, dataset, features, hmm, io, models, scoring  # noqa: E402

GRID = {
    "same_arc_backward_tol_m": [0.0, 5.0, 10.0, 15.0, 25.0],
    "beta_m": [10.0, 30.0, 100.0, 300.0, 1000.0],
    "uturn_penalty": [0.0, 2.0, 5.0, 10.0],
    "lam": [0.0, 0.5],
    "gap_reset_s": [30.0, 60.0],
}
FIXED = {"sigma_m": 8.0, "max_speed_mps": 55.0, "transition_slack_m": 300.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(config.DATA_ROOT))
    parser.add_argument("--dataset", default=str(config.CACHE_DIR / "dataset.npz"))
    parser.add_argument("--oof", default=str(config.CACHE_DIR / "oof_hist_gradient_boosting.npy"))
    parser.add_argument("--artifact-dir", default=str(config.ARTIFACT_DIR))
    parser.add_argument("--tune-trips", type=int, default=24, help="number of train trips used for selection")
    parser.add_argument("--continuity-margin", type=float, default=0.05,
                        help="selected config must beat classifier-only strict continuity by this much")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--out-dir", default=str(config.RESULTS_DIR / "hmm"))
    parser.add_argument("--tag", default="v2", help="output suffix, e.g. v2_refine")
    for key in GRID:
        parser.add_argument(f"--{key.replace('_', '-')}", type=float, nargs="+", default=None,
                            help=f"override the grid values for {key}")
    args = parser.parse_args()
    for key in GRID:
        override = getattr(args, key)
        if override:
            GRID[key] = list(override)
    return args


# ---------------------------------------------------------------------------
# Per-trip precomputation (parameter-independent parts of the decoder)
# ---------------------------------------------------------------------------
class Prepared:
    def __init__(self, trip, cands, probs, labels, confidence):
        self.trip, self.cands, self.labels, self.conf = trip, cands, labels, confidence
        self.probs = np.asarray(probs, dtype=float)
        cfg = hmm.HMMConfig(gap_reset_s=max(GRID["gap_reset_s"]))
        ctx = hmm.DecoderContext(trip, cands, cfg)
        off = cands.offsets
        self.valid = np.array([i for i in range(len(cands)) if off[i + 1] > off[i]], dtype=np.int64)
        L = self.valid.size
        K = int(max((off[i + 1] - off[i]) for i in self.valid)) if L else 1
        self.K, self.L = K, L
        self.k = np.array([off[i + 1] - off[i] for i in self.valid], dtype=np.int64)
        logp = np.full((L, K), -np.inf)
        dist2 = np.zeros((L, K))
        for li, i in enumerate(self.valid):
            lo, hi = int(off[i]), int(off[i + 1])
            p = np.clip(self.probs[lo:hi], 1e-9, None)
            tot = float(p.sum())
            p = p / tot if tot > 0 else np.full(p.size, 1.0 / max(p.size, 1))
            logp[li, : hi - lo] = np.log(p)
            dist2[li, : hi - lo] = cands.distance_m[lo:hi] ** 2
        self.logp, self.dist2 = logp, dist2
        T = max(L - 1, 0)
        self.F = np.full((T, K, K), np.nan, dtype=np.float32)   # same-arc signed progress
        self.D = np.full((T, K, K), np.inf, dtype=np.float32)   # network distance to a different arc
        self.R = np.zeros((T, K, K), dtype=bool)                 # reverse (U-turn) arc
        self.DT = np.zeros(T)
        self.G = np.zeros(T)
        arcs = ctx.arcs
        for t in range(T):
            prev, cur = int(self.valid[t]), int(self.valid[t + 1])
            p_lo, p_hi = int(off[prev]), int(off[prev + 1])
            c_lo, c_hi = int(off[cur]), int(off[cur + 1])
            pa = cands.arc_ids[p_lo:p_hi].astype(np.int64)
            ca = cands.arc_ids[c_lo:c_hi].astype(np.int64)
            pr = np.clip(cands.projection_ratio[p_lo:p_hi], 0.0, 1.0)
            cr = np.clip(cands.projection_ratio[c_lo:c_hi], 0.0, 1.0)
            self.DT[t] = max(float(trip.track.t[cur] - trip.track.t[prev]), 0.0)
            self.G[t] = float(np.linalg.norm(ctx.track_xy[cur] - ctx.track_xy[prev]))
            for i, arc_i in enumerate(pa):
                same = ca == arc_i
                if same.any():
                    self.F[t, i, : ca.size][same] = (cr[same] - pr[i]) * ctx.lengths[arc_i]
                other = ~same
                if other.any():
                    row = ctx.distances_from(int(arcs[arc_i, 1]))
                    if row is not None:
                        tgt = ctx.local_index[arcs[ca[other], 0]]
                        ok = tgt >= 0
                        d = np.full(int(other.sum()), np.inf)
                        d[ok] = row[tgt[ok]] + (1.0 - pr[i]) * ctx.lengths[arc_i] + cr[other][ok] * ctx.lengths[ca[other][ok]]
                        self.D[t, i, : ca.size][other] = d
                self.R[t, i, : ca.size] = (arcs[ca, 0] == arcs[arc_i, 1]) & (arcs[ca, 1] == arcs[arc_i, 0])
        cols = np.arange(K)
        self.colmask = cols[None, :] < self.k[:, None]          # (L, K)
        self.arc_pad = np.full((L, K), -1, dtype=np.int64)
        for li, i in enumerate(self.valid):
            lo, hi = int(off[i]), int(off[i + 1])
            self.arc_pad[li, : hi - lo] = cands.arc_ids[lo:hi]

    # ------------------------------------------------------------------
    def decode(self, cfg: dict) -> np.ndarray:
        n = len(self.cands)
        out = np.full(n, -1, dtype=np.int64)
        if self.L == 0:
            return out
        emis = self.logp - cfg["lam"] * self.dist2 / (2.0 * cfg["sigma_m"] ** 2)
        if self.L > 1:
            rowm = self.colmask[:-1][:, :, None]
            colm = self.colmask[1:][:, None, :]
            tol = max(cfg["same_arc_backward_tol_m"], 0.0)
            same = ~np.isnan(self.F)
            F = np.where(same, self.F, 0.0)
            d = np.where(same, np.where(F >= -tol, np.abs(F), np.inf), self.D)
            maxd = np.maximum(cfg["max_speed_mps"] * self.DT + cfg["transition_slack_m"], 300.0)[:, None, None]
            feas = (d <= maxd) & rowm & colm
            cost = -np.abs(d - self.G[:, None, None]) / cfg["beta_m"] - cfg["uturn_penalty"] * self.R
            la = np.where(feas, cost, -1e9)
            la = np.where(colm, la, -np.inf)
            dead = ~np.any(feas, axis=2, keepdims=True) & rowm
            la = np.where(dead & colm, 0.0, la)
            mx = np.max(np.where(colm, la, -np.inf), axis=2, keepdims=True)
            lse = mx + np.log(np.sum(np.where(colm, np.exp(la - mx), 0.0), axis=2, keepdims=True))
            la = la - lse
            gap = (self.DT >= cfg["gap_reset_s"])[:, None, None]
            uni = -np.log(np.maximum(self.k[1:], 1)).astype(float)[:, None, None]
            la = np.where(gap & colm, np.broadcast_to(uni, la.shape), la)
        delta = emis[0].copy()
        back = np.zeros((self.L, self.K), dtype=np.int64)
        for t in range(1, self.L):
            scores = delta[:, None] + la[t - 1]
            bp = np.argmax(scores, axis=0)
            delta = scores[bp, np.arange(self.K)] + emis[t]
            back[t] = bp
        state = int(np.argmax(delta))
        states = np.empty(self.L, dtype=np.int64)
        for t in range(self.L - 1, -1, -1):
            states[t] = state
            state = int(back[t, state])
        out[self.valid] = self.arc_pad[np.arange(self.L), states]
        return out

    def classifier(self) -> np.ndarray:
        arcs, _ = hmm.classifier_choice(self.cands, self.probs)
        return arcs

    def score(self, arcs: np.ndarray) -> tuple[float, float]:
        lab = self.conf & (self.labels >= 0)
        top1 = float(np.mean(arcs[lab] == self.labels[lab])) if lab.any() else float("nan")
        seq = arcs[arcs >= 0]
        if seq.size:
            seq = seq[np.concatenate(([True], seq[1:] != seq[:-1]))]
        if seq.size < 2:
            return top1, float("nan")
        net = self.trip.network.arcs
        cont = float(np.mean(net[seq[:-1], 1] == net[seq[1:], 0]))
        return top1, cont


def configs():
    keys = list(GRID)
    for vals in itertools.product(*(GRID[k] for k in keys)):
        yield {**FIXED, **dict(zip(keys, vals))}


def main() -> int:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = dataset.load_dataset(args.dataset)
    oof = np.load(args.oof)
    if oof.size != data["y"].size:
        raise SystemExit("OOF probabilities do not match the dataset rows; rebuild with scripts/03_train_models.py")
    trip_ids = [str(t) for t in data["trip_ids"]]
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(len(trip_ids), size=min(args.tune_trips, len(trip_ids)), replace=False).tolist())

    # ---- prepare tuning trips (train, OOF probabilities) ----------------------
    t0 = time.perf_counter()
    tune: list[tuple[str, Prepared]] = []
    for t in chosen:
        trip = io.load_trip(Path(args.data_root) / "train" / trip_ids[t], with_route=False)
        cands = cand_mod.CandidateGenerator(trip.network).generate(trip.track)
        rows = np.flatnonzero(data["row_trip"] == t)
        if rows.size != cands.arc_ids.size or not np.array_equal(data["row_arc"][rows], cands.arc_ids):
            raise SystemExit(f"Candidate sets differ from the cached dataset for trip {trip_ids[t]}")
        obs = np.flatnonzero(data["obs_trip"] == t)
        tune.append((trip_ids[t], Prepared(trip, cands, oof[rows], data["labels"][obs], data["confidence"][obs])))
        print(f"prepared train/{trip_ids[t]} ({len(trip.track)} obs)", flush=True)
    prep_s = time.perf_counter() - t0

    # ---- equivalence check against the real decoder ---------------------------
    small = sorted(tune, key=lambda kv: len(kv[1].cands))[:2]
    for cfg in ({**FIXED, "same_arc_backward_tol_m": 0.0, "beta_m": 10.0, "uturn_penalty": 0.0, "lam": 0.0, "gap_reset_s": 60.0},
                {**FIXED, "same_arc_backward_tol_m": 15.0, "beta_m": 100.0, "uturn_penalty": 5.0, "lam": 0.5, "gap_reset_s": 30.0}):
        for tid, prep in small:
            ref, _ = hmm.decode_hmm(prep.trip, prep.cands, prep.probs, hmm.HMMConfig(**cfg))
            fast = prep.decode(cfg)
            if not np.array_equal(ref, fast):
                raise SystemExit(f"Fast decoder differs from hmm.decode_hmm on {tid} for {cfg}")
    print("fast decoder == hmm.decode_hmm on the check trips/configs", flush=True)

    # ---- grid on tuning trips -------------------------------------------------
    base = [p.score(p.classifier()) for _, p in tune]
    cls_top1 = float(np.nanmean([b[0] for b in base]))
    cls_cont = float(np.nanmean([b[1] for b in base]))
    rows_out = []
    t1 = time.perf_counter()
    for i, cfg in enumerate(configs()):
        sc = [p.score(p.decode(cfg)) for _, p in tune]
        rows_out.append({**cfg, "partition": f"train ({len(tune)} trips, OOF probabilities)",
                         "point_top1": float(np.nanmean([s[0] for s in sc])),
                         "strict_continuity": float(np.nanmean([s[1] for s in sc]))})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} configs ({time.perf_counter() - t1:.0f}s)", flush=True)
    grid = pd.DataFrame(rows_out)
    grid["top1_minus_classifier"] = grid["point_top1"] - cls_top1
    grid["continuity_minus_classifier"] = grid["strict_continuity"] - cls_cont
    grid.sort_values("point_top1", ascending=False).to_csv(out / f"decoder_tuning_{args.tag}.csv", index=False)

    eligible = grid[grid["strict_continuity"] >= cls_cont + args.continuity_margin]
    best = eligible.sort_values(["point_top1", "strict_continuity"], ascending=False).iloc[0]
    best_cfg = {k: float(best[k]) for k in (*GRID, *FIXED)}

    # ---- held-out evaluation on student-test ---------------------------------
    payload = models.load_artifact(Path(args.artifact_dir) / "model.joblib")
    prev_cfg = json.loads((Path(args.artifact_dir) / "hmm_config.json").read_text())
    prev_cfg = {**FIXED, "same_arc_backward_tol_m": 0.0, **prev_cfg}
    eval_rows = []
    for meta in io.discover_trips(args.data_root, partition="student_test"):
        trip = io.load_trip(meta.directory, with_route=False)
        cands = cand_mod.CandidateGenerator(trip.network).generate(trip.track)
        fm = features.build_features(trip, cands, alignment=None)
        probs = payload["model"].predict_proba(fm.X)[:, 1]
        _, al = scoring.load_alignment(meta.directory)
        prep = Prepared(trip, cands, probs, al.arc_ids, al.confidence)
        for name, arcs in (("classifier", prep.classifier()), ("hmm_previous", prep.decode(prev_cfg)),
                           ("hmm_v2", prep.decode(best_cfg))):
            top1, cont = prep.score(arcs)
            eval_rows.append({"trip_id": meta.trip_id, "mode": name, "point_top1": top1, "strict_continuity": cont})
    ev = pd.DataFrame(eval_rows)
    ev.to_csv(out / f"decoder_{args.tag}_student_test.csv", index=False)
    summary_st = ev.groupby("mode")[["point_top1", "strict_continuity"]].mean()

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(["python3", "scripts/05d_tune_hmm_v2.py", *sys.argv[1:]]),
        "selection": f"max train Top-1 s.t. strict continuity >= classifier-only + {args.continuity_margin}",
        "tuning_trips": [tid for tid, _ in tune],
        "grid": GRID, "fixed": FIXED, "n_configs": int(len(grid)), "tag": args.tag,
        "prep_seconds": round(prep_s, 1), "grid_seconds": round(time.perf_counter() - t1, 1),
        "train_classifier_only": {"point_top1": cls_top1, "strict_continuity": cls_cont},
        "train_best": {"point_top1": float(best["point_top1"]), "strict_continuity": float(best["strict_continuity"])},
        "best_config": best_cfg,
        "previous_config": prev_cfg,
        "student_test_mean_per_trip": summary_st.round(4).to_dict(orient="index"),
    }
    with open(out / f"decoder_tuning_{args.tag}_best.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps({k: result[k] for k in ("train_classifier_only", "train_best", "best_config",
                                             "student_test_mean_per_trip")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
