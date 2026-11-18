# PathSnap — viva notes (Group 8)

> **How to read this.** Part I is the Phase 1 journey: what we were given, what
> we built, and what the results told us. Part II is Phase 2: the decisions we
> made when the numbers came in. Part III is the reference material — metrics,
> the numbers ledger, the Q&A bank, the demo and the checklist.
>
> **Rule:** every number is copied from `results/`; keep its path next to it.
> Write `TBD` until it is measured.
>
> **Maintenance:** see §22. Update only the ledger row, paragraph or Q&A row
> your change affects, in the same commit.

---

# Part I — Phase 1: build the vertical slice

Phase 1 answered one question: **can we turn a noisy GPS trace plus a road
network into a defensible, connected route prediction, served live?** Tag
`v0.1-phase1` marks the end of it.

## 1. The task we were given

We get a GPS trace sampled at roughly 1 Hz and the road network around that
trip. The network is a directed graph: `.nodes` and `.arcs`, where **row index
is the ID**. `.route` is the reference arc sequence.

The job is **map matching**: for every observation, decide which road arc the
vehicle was on, then reconstruct a route that is connected in the network.

We chose a two-stage design:

1. **Candidates + classifier** — shortlist ~15 nearby arcs per observation,
   score each pair with 18 trip-relative features, keep the probability.
2. **HMM + Viterbi** — choose the most probable *connected* arc sequence using
   road-network distance and elapsed time.

Why two stages? A per-point classifier is a good judge of one point but cannot
enforce connectivity; a sequence model needs calibrated per-point evidence to
work on. Each stage does what the other cannot.

## 2. First look at the data

| File | Columns | Meaning | Trap |
|---|---|---|---|
| `.track` | `lon, lat, timestamp_s` | Noisy GPS observations | Longitude first; gaps, not a perfect 1 Hz |
| `.nodes` | `lon, lat` | Road-network nodes | Node ID is the row index, valid only in this trip |
| `.arcs` | `from_node, to_node` | Directed segments | Arc ID is the row index; long arcs are chords across curves |
| `.route` | `arc_id` | Reference arc sequence | **Not** one row per GPS point; it must be aligned |

| Partition | Trips | Use |
|---|---|---|
| Train | 80 | Fit, cross-validation, model selection, tuning |
| Student-test | 10 | Development comparison and decoder comparison |
| Evaluator | 10 | Final demo only; never for training, tuning or feature choice |

Three facts shaped every later decision:

1. **Each trip ships its own network.** No shared global graph, so no coordinate,
   node-ID or arc-ID features — everything must be relative to the current trip.
2. **Reference routes are useful but imperfect.** We align, score confidence and
   report flagged trips instead of trusting them blindly.
3. **GPS jitter at 1 Hz is real.** We smooth before deriving heading and speed.

What we measured (`results/eda/`, `docs/DATA_NOTES.md`): 90 trips, 208,417 train
/ 19,160 student-test observations, 0 NaNs and 0 backward timestamps; 106
zero-length arcs, 1,903 duplicate arcs, gaps up to 153,064 s. Two train routes
are unreliable (`00000033`, `00000054`); no student-test route is.

## 3. Making the data machine-ready

We fixed six things before any model saw a row:

1. **Validate** the trip: files, shapes, lon/lat range, monotonic timestamps,
   arc endpoints and route IDs in range. Reject bad trips, do not repair silently.
2. **Project to metres** with a local equirectangular plane around the trip's mean
   latitude: `x = (λ−λ₀)·111320·cos φ₀`, `y = (φ−φ₀)·110540`. Degrees are
   angular; the local plane is accurate at trip scale and fast.
3. **Smooth** positions over a short centred window (default 5), then compute
   heading and speed from smoothed positions and real timestamps.
4. **Generate candidates**: KD-tree over arcs sampled every ~5 m, radius 60 m
   (fallbacks 100/150 m), exact point-to-segment distance, keep top 15.
5. **Align the reference route**: build the route polyline, project points onto
   it, enforce monotonic progress, map arclength back to an arc, flag points
   farther than 30 m as low confidence.
6. **Build features** per (observation, candidate); fit scalers inside training
   folds only; never touch evaluator data.

The order matters: projecting after loading but before smoothing keeps the
window in metres; aligning labels before building features stops a mislabelled
point from becoming a training row.

## 4. Candidates — how we avoided checking 300k arcs

**What we did.** A KD-tree over densely sampled arcs (every ~5 m), radius 60 m
with 100/150 m fallbacks, exact point-to-segment distance on the shortlist, top
15 kept.

**Why sampling and not just endpoints.** Arcs are contracted chords across
curved roads; a long arc's endpoints can be far from the actual road. Dense
samples make the index honest, then exact geometry removes false candidates.

**Result.** Candidate miss rate **0.092% train / 0.032% student-test** at
K=15, ~8.7 candidates per observation, never empty
(`results/eda/candidate_sweep.csv`). The sweep showed K drives recall, not
radius: K=5 ≈1.2%, K=10 ≈0.24%, K=15 ≈0.09%, K=30 ≈0.002%.

**What it told us.** Candidate generation is not our bottleneck. K=15 is the
knee — K=30 buys 0.09 pp for roughly double the candidates. A missing truth is
a hard ceiling on Top-1, so this number gets quoted before any accuracy claim.

## 5. Labels — turning one route into per-point supervision

**What we did.** `.route` is a sequence of arcs, not one label per GPS point. We
built the route polyline, projected each observation onto it, enforced monotonic
progress (the vehicle cannot teleport backwards), mapped arclength back to an
arc, and flagged projections beyond 30 m as low confidence.

**Why flag instead of fix.** A wrong label teaches the classifier the wrong arc.
Reporting coverage is honest; silently repairing bad routes is not.

**Result.** **183,052 / 208,417** train observations labelled; mean per-trip
coverage 96.8%, median 99.87%, pooled 87.8%
(`results/labels/summary.json`, `coverage.csv`). The pooled number is lower than
the per-trip mean because two long trips dominate the counts.

**What it told us.** Labels are good enough to train on, but 12% of train
observations are excluded — and two trips (`00000033`, `00000054`) are flagged
unreliable, so we never quote a per-trip score there without saying why.

## 6. Features — 18 columns, one clear winner

Every feature is computed per (observation, candidate) pair and is trip-relative,
frozen as v1 in `pathsnap/config.py`.

| Group | Examples | Why it helps | Rejected alternative |
|---|---|---|---|
| Geometric | point distance, distance ratio, projection ratio, distance to arc ends | The closest road is usually right — but not at junctions | Raw lon/lat: memorises the city |
| Directional | `abs_heading_diff`, `cos_heading_diff` | Separates parallel and opposite carriageways | Raw 1 Hz bearing: too noisy |
| Topological | arc length, node degrees, reciprocal arc, graph distance from previous best, same-arc flag, graph-vs-GPS residual | Encodes connectivity, which distance cannot | Distance alone: allows teleports |
| Temporal | smoothed speed, gap seconds, candidate count | Rejects impossible movement; relaxes on gaps | Fixed thresholds only |
| Target | reference arc + confidence flag | Supervision without treating the route as truth | Nearest arc to GPS: copies noise |

We ablated the groups (`results/models/feature_ablation.csv`), and the result
was sharper than expected:

- **Heading is the signal.** Removing the two heading columns costs **0.241**
  Top-1 (0.887 → 0.646); nothing else is close.
- **Distance alone (0.567) barely beats the nearest-road baseline (0.575).**
- Relative features mostly improve **calibration** (log loss 0.104 → 0.090),
  which matters because probabilities become HMM emissions.
- Graph/temporal features add only **+0.007** per point — deliberately small,
  because sequence consistency is the HMM's job, not the point classifier's.

**Leakage discipline:** no target-derived feature, no scaler fitted on
validation/test rows, no trip split across folds, no evaluator data, no raw
coordinates or IDs.

## 7. Models — eight families, one selection rule

We trained eight families plus a nearest-road baseline on the same candidate
sets, labels and GroupKFold(5) splits over the 80 train trips (seed 42).
GroupKFold is not optional: observations within a trip are correlated, so random
folds leak trip-specific patterns.

| Model | CV Top-1 | Fold std | Macro F1 | Log loss | Fit s (5 folds) |
|---|---|---|---|---|---|
| **HistGradientBoosting** (selected) | **0.887** | 0.027 | **0.928** | **0.084** | 27 |
| Random forest | 0.886 | 0.026 | 0.923 | 0.098 | 286 |
| Extra Trees | 0.885 | 0.025 | 0.917 | 0.115 | 138 |
| Decision tree | 0.879 | 0.024 | 0.898 | 0.142 | 21 |
| MLP | 0.878 | 0.028 | 0.922 | 0.096 | 170* |
| Linear SVM (calibrated) | 0.871 | 0.025 | 0.898 | 0.123 | 21* |
| Logistic regression | 0.871 | 0.025 | 0.857 | 0.179 | 2 |
| Gaussian NB | 0.845 | 0.032 | 0.137 | 4.746 | 0.3 |
| Nearest-road baseline | 0.575 | 0.020 | — | — | — |

\* Fit times measured while another job shared the CPU; upper bounds. Source:
`results/models/model_comparison.csv`, run `20261118T052120Z`.

**Selection rule, fixed before reading the table:** highest pooled out-of-fold
observation Top-1; tie-break macro F1, then log loss; guardrails positive-class
F1 and cost. No evaluator data.

**Why HistGradientBoosting:**

- Highest CV Top-1 **and** best on both guardrails (macro F1 0.928, log loss 0.084).
- 10× cheaper to fit than the random forest (27 s vs 286 s).
- Best-calibrated probabilities of the strong models — those become the HMM
  emissions, so calibration is functional, not cosmetic.
- The RF is 0.0003 behind, inside fold noise (σ ≈ 0.027), so this is **not** a
  significant accuracy win. The tie-break is guardrails and cost. Say that
  honestly.

**What the rest of the table taught us:**

- **Linear models and NB underfit.** Near-zero train/validation gap, but a linear
  boundary cannot express the heading × distance interaction. NB still ranks
  reasonably (0.845) yet its log loss is 4.7 because correlated features are
  treated as independent — ranking is not calibration.
- **Decision tree and RF overfit most.** RF reaches 0.939 on training trips vs
  0.886 held out; HGB sits between (0.923 vs 0.887) thanks to early stopping and
  `min_samples_leaf=40`.
- **The baseline is the honest floor.** Nearest-road scores 0.575; distance-only
  ablation 0.567. Distance barely beats snapping; heading is the model.

## 8. HMM — adding connectivity the classifier cannot see

**Why.** The classifier scores candidates independently, so consecutive
predictions can land on disconnected arcs — teleports. The HMM adds sequence
context and network feasibility.

**How.** The hidden state is the candidate arc at observation `t`. Emissions are
normalised classifier probabilities, optionally distance-blended. Transitions
compare network travel distance with GPS displacement:
`log a_ij = −|d_net − d_gps| / β − γ·1[U-turn]`, with infeasible pairs at
`−∞`. Viterbi runs the usual recurrence in log space,
`δ_t(j) = log ẽ_t(j) + max_i[δ_{t−1}(i) + log a_ij]`, with backpointers.

**Implementation detail that matters.** We build a directed corridor CSR graph
per trip and cache bounded Dijkstra rows per source node. Same-arc motion uses
forward projection progress; empty candidate layers stay unmatched; large gaps
reset the chain rather than pretending connectivity.

**Result.** Synthetic transition tests pass (`tests/test_hmm.py`). Real-trip
warm transition latency: **5.17 s** for 19,313 observations (0.268 ms per
transition) and **4.62 s** for 16,031 observations (0.288 ms) with defaults
(`results/hmm/transition_benchmark.csv`).

**What it told us.** The decoder is feasible on full trips. Whether it *helps*
accuracy is a tuning question, not a plumbing question — which is exactly what
Phase 2 had to answer.

## 9. Serving it: API, UI, and a ten-trip integration

**What we built.** FastAPI on Laptop 1 (`/health`, `/version`, `/trips`,
`/predict`) and a thin Streamlit client on Laptop 2. The prediction path loads
trips with `with_route=False`; `pathsnap/scoring.py` opens `.route` only after
the prediction exists. The response never contains the reference route or
per-observation labels.

**Result.** All **10/10** student-test trips ran in both decoder modes
(`results/integration/run_log.csv`):

| Metric | Classifier-only | HMM (defaults) |
|---|---|---|
| Observation Top-1 | 0.897 | 0.851 |
| Route continuity | 0.756 | **0.839** |
| Sequence LCS F1 | 0.763 | 0.729 |
| Latency p50 / p95 | 438 / 1179 ms | 1213 / 3579 ms |

Worst live API call: **4.07 s** on 3,958 points (`00000056`, HMM). The HMM
raised continuity on 10/10 trips but lowered Top-1 on 9/10 — with default
parameters.

**What it told us.** The platform is live and server-scored. The open risk is
the decoder configuration, not the API/UI. That finding set the Phase 2 agenda.

## 10. Phase 1 in one line

We proved the pipeline end to end: candidates are not the bottleneck (0.092%
miss), labels are usable (183k observations), heading is the feature that
matters, HGB is the best classifier, and the live two-process system serves
10/10 trips. The decoder was implemented but untuned.

---

# Part II — Phase 2: decide, tune, defend

Phase 2 is not "more building". It is the part where the numbers decide what we
keep, and where the demo gets made real.

## 11. Model freeze: we ran Optuna and kept the default

**What we did.** 40 TPE trials over a bounded HistGradientBoosting space,
GroupKFold(5) by trip, objective = pooled OOF observation Top-1, 34.9 minutes
wall clock (`results/optuna/`).

**Result.** The best trial reached **0.8872** vs **0.8868** untuned — a +0.0004
gain, about 1/65 of the fold standard deviation. Macro F1 and log loss were
slightly *worse*, and on the 10 student-test trips the tuned model scored
**0.8775** vs **0.8818** untuned.

**Decision.** Keep the untuned model. `results/models/tuned_vs_untuned_decision.json`
records it, and the artifact stays `6c229c610200`.

**Why this is the right call.** A gain inside fold noise that does not survive
held-out trips is not a gain; it is search variance. The lesson is worth saying
aloud in the viva: the limit here is per-point ambiguity, not hyperparameters.
Reporting the negative result is stronger than hiding it.

## 12. Student-test: the model holds up, with one caveat

The selected model scores **Top-1 0.882, macro F1 0.925, log loss 0.078** on the
10 unseen student-test trips (`results/models/student_test_comparison.csv`), and
is still best of the eight families.

The caveat: linear models lose ≈2.6 points from CV to student-test, while HGB
loses 0.5. That gap is the generalisation story — and it is also why we never
quote CV alone.

## 13. Decoder tuning: the open Phase 2 question

Default HMM raises continuity but costs Top-1. B's T17 review found a concrete
suspect: the same-arc rule forbids *any* backward progress, yet GPS jitter moves
the projection backwards in **18.8%** of true same-arc steps (median 3.4 m, p95
14.3 m; 80 train trips, `results/models/hmm_review_same_arc.json`). At stops and
in slow traffic that can forbid the true transition.

The planned test (T25): a 10–15 m backward tolerance plus λ/σ/β tuning, then the
same classifier-vs-HMM comparison on student-test. Until those rows exist in
`results/hmm/`, HMM accuracy numbers stay `TBD` and the integration comparison
stays labelled **preliminary**.

## 14. Deployment: what is proven and what is not

Proven: two-process loopback, live inference, server-side scoring, 10/10 trips,
screenshots and API logs, and a dark evaluator UI with an explicit fallback
banner.

Not yet proven: the physical two-laptop path (LAN/hotspot, firewall, `/health`
from Laptop 2) and the ten reserved evaluator trips, which are not mounted
locally. Both are rehearsal tasks, not code tasks.

## 15. Where we are

| Area | State | Evidence |
|---|---|---|
| Data, candidates, labels | ✅ Phase 1 | `results/eda/`, `results/labels/` |
| Features + ablation | ✅ Phase 1 | `results/models/feature_ablation.csv` |
| Model comparison + freeze | ✅ Phase 2 | `results/models/`, `results/optuna/` |
| HMM implementation | ✅ Phase 1 | `tests/test_hmm.py`, `results/hmm/transition_benchmark.csv` |
| HMM tuning + comparison | ⬜ Phase 2 (C) | `results/hmm/` |
| API/UI/integration | ✅ Phase 1 | `results/integration/`, `results/screenshots/` |
| Two-laptop rehearsal | 🟨 local only | `docs/DEPLOYMENT.md` |
| Report sections | 🟨 §3–4 done, others pending | `report/` |

---

# Part III — Reference

## 16. Metrics

| Metric | Definition | Why |
|---|---|---|
| Observation Top-1 | argmax candidate equals the aligned reference arc | Direct map-matching quality |
| Positive F1 / macro F1 | Candidate-level precision/recall balance | Class imbalance |
| Log loss | Probability quality | HMM emissions need calibration |
| Candidate miss rate | Share of truths absent from `C_t` | Hard ceiling on Top-1 |
| Route continuity | Share of consecutive predicted arcs that are connected | Penalises teleports |
| Sequence LCS F1 | Order-aware F1 vs the reference arc sequence | Route-level quality |
| Exact sequence match | Predicted sequence equals reference | Strict diagnostic |
| Inference ms | Per stage and per trip | Deployment feasibility |

Primary: CV observation Top-1 on identical candidate sets. Guardrails: positive
F1, log loss, fold stability. Final gate: continuity and sequence F1 on
student-test. Not accuracy — candidate pairs are ~90% negative.

## 17. Numbers ledger (single source of truth)

| Claim | Value | Source | Owner | Status |
|---|---|---|---|---|
| Trips in train / student-test | 80 / 10 | `results/eda/inventory.csv` | A | verified |
| Train / student-test observations | 208,417 / 19,160 | `results/eda/summary.json` | A | verified |
| Candidate miss rate @ 60 m | 0.092% train / 0.032% student-test (top-K 15, 8.7 candidates/obs) | `results/eda/candidate_sweep.csv` | A | verified |
| Label coverage (train) | mean 96.8% per trip, median 99.87%, pooled 87.8% | `results/labels/summary.json` | A | verified |
| Labelled observations (train) | 183,052 / 208,417 | `results/labels/summary.json` | A | verified |
| Miss rate vs top-K (train) | K=5 ≈1.2%, K=10 ≈0.24%, K=15 ≈0.09%, K=30 ≈0.002% | `results/eda/candidate_sweep.csv` | A | verified |
| Unreliable reference trips | 2 (`00000033`, `00000054`); 0 in student-test | `results/labels/coverage.csv` | A | verified |
| Dataset candidate rows (train) | 1,732,459 (182,884 pos / 1,402,460 neg / 147,115 ignored) | `results/models/dataset_summary.json` | B | verified |
| Best CV Top-1 | 0.887 HistGradientBoosting (fold std 0.027); baseline 0.575 | `results/models/model_comparison.csv` | B | verified |
| Best CV macro F1 / log loss | 0.928 / 0.084 HistGradientBoosting | `results/models/model_comparison.csv` | B | verified |
| Student-test Top-1 (selected) | 0.882 HistGradientBoosting (macro F1 0.925); best of 8 | `results/models/student_test_comparison.csv` | B | verified |
| Student-test log loss | 0.078 HistGradientBoosting | `results/models/student_test_comparison.csv` | B | verified |
| Best model + config | HistGradientBoosting, default params (kept after Optuna); artifact `6c229c610200` | `results/models/tuned_vs_untuned_decision.json` | B | verified |
| Optuna trials / best CV score | 40 TPE trials, GroupKFold(5), 34.9 min; best 0.8872 (trial 18) vs 0.8868 untuned | `results/optuna/best_params.json` | B | verified |
| Tuned vs untuned delta | CV +0.0004 Top-1 (guardrails worse); student-test −0.0043 → keep untuned | `results/models/tuned_vs_untuned.csv` | B | verified |
| Decoder best (λ, σ, β, gap) | TBD | `results/hmm/decoder_tuning.csv` | C | TBD |
| HMM Top-1 vs classifier-only | TBD | `results/hmm/classifier_vs_hmm.csv` | C | TBD |
| HMM continuity vs classifier-only | TBD | `results/hmm/classifier_vs_hmm.csv` | C | TBD |
| HMM sequence LCS F1 | TBD | `results/hmm/classifier_vs_hmm.csv` | C | TBD |
| HMM warm transition latency | 5.17 s for 19,313 obs (0.268 ms/transition); 4.62 s for 16,031 obs (0.288 ms) | `results/hmm/transition_benchmark.csv` | C | verified (Phase 1, defaults) |
| Integration HMM vs classifier (preliminary) | Top-1 0.851 vs 0.897; continuity 0.839 vs 0.756; LCS F1 0.729 vs 0.763 | `results/integration/per_trip_comparison.csv` | D → C | measured; C to tune |
| Same-arc backward jitter | 18.8% of true same-arc steps; median 3.4 m, p95 14.3 m | `results/models/hmm_review_same_arc.json` | B → C | verified |
| API p50 / p95 latency | HMM 1213 / 3579 ms; classifier 438 / 1179 ms | `results/integration/stage_timings.csv` | D | verified |
| Student-test trips passing | 10/10 both modes; worst live API 4.07 s on 3,958 points | `results/integration/run_log.csv` | D | verified |
| Model artifact version | `6c229c610200` (untuned; kept after Optuna) | `artifacts/model_card.json` | D | verified |

## 18. Q&A bank

### 18.1 Rapid-fire one-liners

| Question | Answer |
|---|---|
| What is map matching? | Assigning noisy GPS observations to road-network segments and reconstructing a connected route. |
| Why is it hard? | Noise, low sampling, gaps, parallel roads, junctions, one-way rules, long chords. |
| What is an emission? | The likelihood of an observation given a hidden road segment. |
| What is a transition? | The likelihood of moving between two road segments, constrained by the network and time. |
| Why log space? | Many small probabilities underflow when multiplied. |
| Complexity of Viterbi? | `O(T·K²)` transitions plus shortest-path computation. |
| What is candidate miss rate? | The fraction of truths absent from the candidate set. |
| What is route continuity? | The fraction of consecutive predictions that are connected in the road graph. |
| What is LCS F1? | Order-aware precision/recall of the predicted arc sequence against the reference. |
| Why keep ground truth server-side? | Integrity: the model and the UI must not see the answer. |
| What is the model input? | A vector of relative geometric/topological features for one observation–candidate pair. |
| What is the model output? | A calibrated probability that the candidate is the correct arc. |
| What does the HMM add? | Sequence consistency and network feasibility. |
| What is the biggest failure mode? | A truth arc missing from the candidate set, or a misaligned reference route. |

### 18.2 Questions that get asked

| Question | Answer |
|---|---|
| Why not snap to the nearest road? | That is our baseline (0.575); it fails on parallel roads and at junctions. |
| Why a local projection? | Degrees are angular; a tangent plane at trip scale is accurate and fast. |
| Why smooth? | 1 Hz jitter makes raw headings and speeds unreliable; it uses no labels. |
| Why candidates? | Networks have up to 300k arcs; a shortlist is tractable and its recall is measurable. |
| What is the miss rate and why care? | 0.092% train / 0.032% student-test at K=15; missing truths cap Top-1. |
| How are labels built? | Project onto the route polyline, enforce monotonic progress, map to an arc, flag low confidence. |
| Why monotonic progress? | The vehicle cannot teleport backwards along the route; it removes ambiguity. |
| Why no coordinate or ID features? | They are trip-local; the model would memorise training cities. |
| Why GroupKFold? | Observations in a trip are correlated; random folds leak trip-specific patterns. |
| Why not accuracy? | Candidate pairs are imbalanced; predicting "negative" always scores high. |
| Why HistGradientBoosting? | Best Top-1 plus guardrails, fast, well calibrated. Gap to RF is inside fold noise. |
| Why calibrate probabilities? | They become HMM emissions, and log loss must mean something. |
| Write the Viterbi recurrence. | `δ_t(j) = log ẽ_t(j) + max_i[δ_{t−1}(i) + log a_ij]`, backpointers, log space. |
| Did tuning help? | Barely: 40 TPE trials, best CV 0.8872 vs 0.8868 (+0.0004, ~1/65 of fold std), worse guardrails and lower student-test, so we kept the default. The limit is per-point ambiguity, not hyperparameters. |
| Did the HMM help? | It raises continuity but currently costs Top-1 with default parameters; B's review points at backward same-arc jitter (18.8% of steps) and T25 will test a 10–15 m tolerance. |
| How is ground truth kept server-side? | Prediction loads trips without `.route`; scoring runs afterwards and only metrics leave the API. |
| How do you prove the API is live? | The per-request log line with trip, latency and model version, plus the UI screenshot. |

### 18.3 Who must be able to explain what

| Member | Must know without notes |
|---|---|
| A | File semantics, inventory, projection, candidates, label alignment, flagged trips, maps |
| B | Features and ablation, CV protocol, metrics, model selection, Optuna and the freeze decision |
| C | HMM states/emissions/transitions, Viterbi, gaps, decoder tuning and comparison |
| D | API contract, ground-truth isolation, two-laptop deployment, logging evidence, demo |

Two honest failure cases to have ready: a parallel-road confusion and a large GPS
gap (before/after map or metric).

## 19. Demo script (5–7 minutes)

1. **Context (30 s):** noisy GPS, recover the road sequence, UI never sees truth.
2. **UI (30 s):** health badge, trip list, pick a trip.
3. **Run (30 s):** click Run; point to the spinner and the API log line.
4. **Results (60 s):** metric cards, predicted route on the map, observed GPS.
5. **Journey (90 s):** candidates → labels → features → classifier → HMM → API.
6. **Evidence (60 s):** model table, Optuna result and why we kept the default,
   classifier-vs-HMM, one failure.
7. **Honesty (30 s):** miss rate, flagged routes, untuned decoder, what is next.

Show the system first, not slides.

## 20. Final checklist (T-60 minutes)

- [ ] `git status` clean; `make test` and `make smoke` pass.
- [ ] No `data/`, `.env`, secrets or course PDFs committed; repo public.
- [ ] Artifact, feature config and HMM config load; metrics match the report.
- [ ] All ten trips run through the API/UI without manual intervention.
- [ ] Ground-truth isolation test passes; no reference route in any response.
- [ ] API logs visible; model version printed; screenshots saved.
- [ ] Two laptops on one network; `curl /health` works from the client.
- [ ] Demo rehearsed end to end at least once.
- [ ] No `TBD` left for numbers that appear in the report.
- [ ] Every member can explain their section and one they do not own.

## 21. AI-assistance disclosure (draft)

> AI tools were used for repository scaffolding, code-review suggestions,
> debugging and language editing. All training, tuning, decoding, measurements
> and visualisations were executed by the team. Every member can explain the
> submitted code, results and decisions. No reported metric was generated
> without an artifact in the repository.

## 22. Keeping these notes current

This file is short on purpose, so it is cheap to keep true. Four rules:

1. **Numbers.** When a number changes, update its row in the ledger (§17) and any
   sentence that quotes it. Quote from the ledger, never keep a second copy.
2. **Behaviour.** If the pipeline, features, model choice, decoder or API contract
   changes, edit the one paragraph that describes it.
3. **Questions.** When someone is asked something we could not answer, add it to
   §18 with a one-line answer. That is how the bank grows.
4. **Ownership.** Whoever changes the behaviour updates this file in the same
   commit. A compiles it before a mock viva; D checks the demo evidence.

Audit before a milestone: `grep -n TBD viva.md` — every remaining `TBD` must be
planned Phase 2 work, not a forgotten number.
