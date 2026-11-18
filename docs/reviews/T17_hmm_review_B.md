# T17 review — HMM design note vs code (reviewer: B)

- **Reviewed:** `docs/HMM_DESIGN.md` and `pathsnap/hmm.py` at `main` @ `3bb24a7`
- **Reviewer:** B (classifier owner; the HMM consumes B's probabilities)
- **Date:** 2026-11-18
- **Relation to D's cross-review:** D cross-reviewed T17 for B (`docs/HMM_DESIGN.md` §11)
  and documented the 300 m floor and 5 km Dijkstra bound. This review agrees on the
  formulas and adds a behavioural measurement on real trips (finding 1) that a
  formula check cannot reveal.
- **Verdict:** ✅ **Approved for Phase 1.** The formulas in the design note match
  the code, and the classifier → HMM interface is correct. One behavioural
  issue (finding 1) is significant enough that C should address it during
  decoder tuning (T25); it is not a Phase-1 blocker.

## 1. Formula ↔ code check

| Design note | Code (`pathsnap/hmm.py`) | Match |
|---|---|---|
| States = candidate occurrences per observation (§2) | `viterbi_layers` over per-observation candidate layers; `decode_hmm` maps state index → `candidates.arc_ids[lo + state]` | ✅ |
| Emission `log(p_j / Σ_k p_k) − λ·d²/(2σ²)`, uniform if all zero (§3) | `emission_scores` L61–72: clip 1e-9, normalise, `log(p) − lam·dist²/(2·sigma²)` | ✅ |
| `d_net = exit_offset + shortest_path(to_i → from_j) + entry_offset` (§4) | `transition` L137–158 | ✅ |
| Infeasible if no path or `d_net > max_speed·dt + slack` (§4) | `max_d = max(max_speed·dt + slack, 300)` L134; `feasible = d <= max_d` | ✅ (the `max(…, 300)` floor is redundant with slack 300; harmless) |
| `log a_ij = −|d_net − gps_disp|/β − uturn·[reverse]` (§4) | L164–169 | ✅ |
| Row log-sum-exp normalisation; weak uniform fallback for dead rows (§4) | L171–178 | ✅ |
| Same arc: forward progress only, backwards infeasible (§4) | L143–145 | ✅ matches — but see finding 1 |
| Gap `dt ≥ gap_reset_s` → log-uniform transition (§6) | L128–131 | ✅ |
| Empty candidate layer skipped, output `−1` (§6) | `decode_hmm` `valid` list L213–221 | ✅ |
| Viterbi `delta_1 = log e_1 + log start_prior` (§5) | `viterbi_layers` starts from `log_emissions[0]` only | ⚠️ equivalent to a **uniform** start prior; the note should say so |

## 2. Classifier → HMM interface (B's side)

- The API passes `estimator.predict_proba(X)[:, 1]` in candidate order
  (`backend/service.py`), and the artifact loader rejects a feature list that
  differs from `config.FEATURE_NAMES`. ✅
- The selected model (HistGradientBoosting) has the best CV log loss (0.084,
  `results/models/model_comparison.csv`), which is the property the emission
  depends on. ✅
- **Probability scale:** the classifier is fit with 3 negatives per positive
  (`subsample_negatives`), while real candidate sets have ≈ 7.7 negatives per
  positive (`results/models/dataset_summary.json`). Absolute probabilities are
  therefore inflated by a roughly constant odds factor. Per-observation
  normalisation keeps the ranking, and λ/β tuning on the *actual* artifact
  absorbs the remaining scale effect, so no code change is required — but T25
  must tune against the exported artifact, and re-check after B's Optuna
  re-export (T23).

## 3. Findings

### Finding 1 — same-arc backward jitter is treated as infeasible (significant)

Any backward projection progress on the same arc gets `d = ∞` (L145), so the
true "stayed on this road" transition is forbidden whenever raw GPS jitter
moves the projection slightly backwards (typically while slow or stopped).

Measured on all 80 train trips (`python3 scripts/review_hmm_same_arc.py`,
evidence `results/models/hmm_review_same_arc.{json,csv}`): of 148,039
consecutive observation pairs where the reference says the vehicle stayed on
the same arc,

| Backward progress | Share of true same-arc steps |
|---|---|
| > 0 m (forbidden by the current code) | **18.8%** |
| > 2 m | 12.8% |
| > 5 m | 6.6% |
| > 10 m | 2.0% |
| > 15 m | 0.85% |

Median backward step 3.4 m, p95 14.3 m — consistent with GPS noise, not real
reversing. When the true transition is forbidden, Viterbi must switch to
another candidate (often the reverse arc with a U-turn penalty) or hit the
dead-row fallback, which costs Top-1 and continuity, especially at stops.

Caveat: the worst trips (`00000011` 43.5%, `00000002` 42.0%) are
out-and-back/loop routes, where part of the "backward" share may come from
nearest-arc labels picking the opposite-direction arc (A's open follow-up in
`docs/STATUS.md`). The median trip is still well above zero, so the effect is
not only a label artefact.

**Corroborating evidence:** D's preliminary 10-trip student-test integration
(`results/integration/per_trip_comparison.csv`, default HMM config, untuned
artifact) shows the HMM *below* classifier-only on point Top-1 (0.851 vs
0.897) and LCS F1 (0.729 vs 0.763), while improving continuity (0.839 vs
0.756). Forbidden same-arc transitions are a plausible contributor to the
Top-1 loss; this is a hypothesis for C to test, not a measured cause.

**Recommendation for C (T25):** allow a small backward tolerance on the same
arc, e.g. treat `forward ≥ −τ` as feasible with `d = |forward|` (so the
`|d − gps_disp|/β` cost still penalises it), and tune `τ` ∈ {0, 5, 10, 15} m
alongside λ, σ, β. At τ = 15 m, 99.1% of true same-arc steps become feasible.

### Finding 2 — Dijkstra cap must cover the tuned gap reset (latent)

`distances_from` caps shortest paths at 5,000 m (L112) while
`max_d = 55·dt + 300`. With the default 30 s gap reset `max_d ≤ 1,950 m`, so
this is fine today. If T25 raises `gap_reset_s` above ≈ 85 s, legitimate
transitions longer than 5 km would be wrongly infeasible. Derive the cap from
`max_speed_mps · gap_reset_s + transition_slack_m` instead of a constant.

### Finding 3 — distance is counted twice (tuning note)

The classifier already uses `log_distance`, `distance_ratio_best` and
`candidate_rank`; the emission adds `−λ·d²/(2σ²)` on top. That is a legitimate
modelling choice, but the T25 grid should include **λ = 0** so the report can
show whether the extra distance term helps.

### Finding 4 — design-note wording (minor)

§5 mentions `log start_prior(j)`; the code uses none (uniform). Update the
note to say "uniform start prior".

## 4. Disposition

| Item | Owner | When | Blocking Phase 1? |
|---|---|---|---|
| Finding 1: same-arc tolerance τ | C | T25 | No |
| Finding 2: Dijkstra cap from config | C | T25 | No |
| Finding 3: λ = 0 in grid | C | T25 | No |
| Finding 4: uniform start prior wording | C | next doc edit | No |
| Re-tune after B's tuned artifact | C (after B T23) | Phase 2 | No |

T17 Definition of Done ("reviewed by B; formulas match code") is met.
