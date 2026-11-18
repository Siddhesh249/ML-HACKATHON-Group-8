# PathSnap — viva notes (Group 8)

> **Purpose:** short, current notes for the viva and the demo. Explain *why*
> first, then the number. This is not a report and not a session log.
>
> **Rule:** every number below is copied from `results/`; keep its path next to
> it. Write `TBD` until it is measured.
>
> **Status:** Phase 1 complete (T10–T21, tag `v0.1-phase1`). Data, labels,
> candidates, features, 8 model families, the HMM implementation, API/UI and a
> 10-trip integration are measured. Optuna, decoder tuning and the tuned
> artifact are Phase 2.
>
> **Maintenance:** see §14. Whoever changes a number or a behaviour updates this
> file in the same commit.

---

## 1. The task in one paragraph

We are given a noisy GPS trace (~1 Hz) and the road network around that trip.
The network is a directed graph: `.nodes` and `.arcs`, where **row index = ID**.
`.route` is the reference arc sequence. PathSnap does map matching in two stages:

1. **Candidates + classifier** — shortlist ~15 nearby arcs per observation,
   score each pair with 18 trip-relative features, keep the probability.
2. **HMM + Viterbi** — choose the most probable *connected* arc sequence using
   road-network distance and elapsed time.

We train on 80 trips with GroupKFold by trip, compare on 10 student-test trips,
and keep the 10 evaluator trips for the final live demo only. Ground truth stays
on the server.

---

## 2. The data

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

Three facts that shape the design:

1. **Each trip ships its own network.** No shared global graph, so no coordinate,
   node-ID or arc-ID features — everything is relative to the trip.
2. **Reference routes are useful but imperfect.** We align, score confidence and
   report flagged trips instead of trusting them blindly.
3. **GPS jitter at 1 Hz is real.** We smooth before deriving heading and speed.

Measured data quality (`results/eda/`, `docs/DATA_NOTES.md`): 90 trips,
208,417 train / 19,160 student-test observations, 0 NaNs and 0 backward
timestamps; 106 zero-length arcs, 1,903 duplicate arcs, gaps up to 153,064 s.
Two train routes are unreliable (`00000033`, `00000054`).

---

## 3. Preprocessing — six steps

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

---

## 4. Features — why each group

18 trip-relative features, frozen in `pathsnap/config.py` (v1):

| Group | Examples | Why it helps | Rejected alternative |
|---|---|---|---|
| Geometric | point distance, distance ratio, projection ratio, distance to arc ends | The closest road is usually right — but not at junctions | Raw lon/lat: memorises the city |
| Directional | `abs_heading_diff`, `cos_heading_diff` | Separates parallel and opposite carriageways | Raw 1 Hz bearing: too noisy |
| Topological | arc length, node degrees, reciprocal arc, graph distance from previous best, same-arc flag, graph-vs-GPS residual | Encodes connectivity, which distance cannot | Distance alone: allows teleports |
| Temporal | smoothed speed, gap seconds, candidate count | Rejects impossible movement; relaxes on gaps | Fixed thresholds only |
| Target | reference arc + confidence flag | Supervision without treating the route as truth | Nearest arc to GPS: copies noise |

Measured importance (`results/models/`) — the honest summary:

- **Heading is the signal.** Removing the two heading columns costs **0.24**
  Top-1; nothing else comes close.
- **Distance alone (0.567) is barely the nearest-road baseline (0.575).**
- Relative features mostly improve **calibration** (log loss 0.104 → 0.090),
  which matters because probabilities become HMM emissions.
- Graph/temporal features add **+0.007** per point — small by design; sequence
  consistency is the HMM's job.

Leakage rules: no target-derived feature, no scaler fitted on validation/test,
no trip split across folds, no evaluator data, no raw coordinates or IDs.

---

## 5. Model comparison and selection

GroupKFold(5) by trip on the 80 training trips, seed 42, identical candidate sets
and labels for every row. Source: `results/models/model_comparison.csv`, run
`20261118T052120Z`. This table is the evidence for the model choice.

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

\* Fit times measured while another job shared the CPU; upper bounds.

**Selection rule, fixed before reading the table:** highest pooled out-of-fold
observation Top-1; tie-break macro F1, then log loss; guardrails positive-class
F1 and cost. No evaluator data at any point.

**Why HistGradientBoosting:**

- Highest CV Top-1 (0.887) **and** best on both guardrails (macro F1 0.928,
  log loss 0.084).
- 10× cheaper to fit than the random forest (27 s vs 286 s), with fewer
  candidates needing a second pass at inference.
- Best-calibrated probabilities of the strong models — and those probabilities
  become the HMM emissions, so calibration is a functional requirement here.
- The random forest is 0.0003 behind, which is inside fold noise (std ≈ 0.027).
  The tie-break is therefore guardrails + cost, **not** a significant accuracy
  win. Say that honestly.

**What the rest of the table tells us:**

- **Linear models and NB underfit.** Near-zero train/validation gap, but a linear
  boundary cannot express the heading × distance interaction. NB still ranks
  reasonably (0.845) yet its log loss is 4.7 because correlated features are
  treated as independent — ranking is not calibration.
- **Decision tree and RF overfit most.** RF reaches 0.939 on training trips vs
  0.886 held out; unlimited depth memorises trip-specific patterns. HGB sits
  between (0.923 vs 0.887) thanks to early stopping and `min_samples_leaf=40`.
- **The baseline is the honest floor.** Nearest-road scores 0.575, and the
  distance-only ablation scores 0.567. Distance barely beats snapping; heading
  is what makes the classifier work.

Not yet measured: student-test numbers and tuned-vs-untuned (Phase 2, T24).

---

## 6. HMM and Viterbi (short version)

The classifier scores candidates independently and can jump between disconnected
roads. The HMM adds sequence context and network feasibility.

- **State:** the candidate arc at observation `t` (candidate set `C_t`, not all arcs).
- **Emission:** normalised classifier probability, optionally blended with a
  distance term `−λ·d²/(2σ²)`.
- **Transition:** compare network travel distance with GPS displacement:
  `log a_ij = −|d_net − d_gps| / β − γ·1[U-turn]`; infeasible pairs get `−∞`.
- **Recurrence:** `δ_t(j) = log ẽ_t(j) + max_i [δ_{t−1}(i) + log a_ij]`,
  with backpointers and backtracking from the best final state. Log space, `O(TK²)`.

Gaps: skip the state layer and mark the observation unmatched; large time gaps
reset the chain. Missing candidates are a hard ceiling — the decoder cannot
recover a truth arc that is not in `C_t`.

Implementation: directed corridor CSR graph with cached bounded Dijkstra rows per
source node; same-arc motion uses forward projection progress; empty layers stay
unmatched. Covered by `tests/test_hmm.py`. Real-trip benchmark:
`python -m scripts.benchmark_hmm` → 5.17 s of warm transitions on the largest
train trip (19,313 observations, 0.268 ms per transition) and 4.62 s on
`00000084` (0.288 ms), both with default parameters
(`results/hmm/transition_benchmark.csv`).

---

## 7. Metrics

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

---

## 8. API, UI and ground-truth isolation

| Endpoint | Purpose |
|---|---|
| `GET /health` | Ready? model name/version, trip count, partition |
| `GET /version` | API version, git commit, Python, model version |
| `GET /trips` | The ten trips (metadata only) |
| `POST /predict` | Run the pipeline, score server-side, return route + metrics + timings |

`POST /predict` takes `{"trip_id": "...", "mode": "hmm"}`; `mode` is `hmm` or
`classifier` for the report comparison. The response contains the predicted
route, observed points, aggregate metrics and stage timings — **never** the
reference route or per-observation labels.

Isolation: the predictor loads trips with `with_route=False`; `pathsnap/scoring.py`
opens `.route` only after prediction. Laptop 1 hosts model, decoder and scoring;
Laptop 2 runs a thin Streamlit client (`PATHSNAP_API_URL`) with a trip selector,
metric cards, stage timings and a map with a predicted-route layer and legend.

The UI must not load `.route`, compute metrics client-side, import the model or
display the reference route.

---

## 9. Numbers ledger (single source of truth)

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
| Best CV Top-1 | 0.887 HistGradientBoosting (fold std 0.027); baseline 0.575 | `results/models/model_comparison.csv` | B | verified (untuned) |
| Best CV macro F1 / log loss | 0.928 / 0.084 HistGradientBoosting | `results/models/model_comparison.csv` | B | verified (untuned) |
| Student-test Top-1 (selected) | TBD | `results/models/model_comparison.csv` | B | TBD |
| Student-test log loss | TBD | `results/models/model_comparison.csv` | B | TBD |
| Best model + config | HistGradientBoosting, default params; preliminary | `results/models/selected_model.json` | B | verified (preliminary) |
| Optuna trials / best CV score | TBD | `results/optuna/trials.csv` | B | TBD |
| Tuned vs untuned delta | TBD | `results/models/tuned_vs_untuned.csv` | B | TBD |
| Decoder best (λ, σ, β, gap) | TBD | `results/hmm/decoder_tuning.csv` | C | TBD |
| HMM Top-1 vs classifier-only | TBD | `results/hmm/classifier_vs_hmm.csv` | C | TBD |
| HMM continuity vs classifier-only | TBD | `results/hmm/classifier_vs_hmm.csv` | C | TBD |
| HMM sequence LCS F1 | TBD | `results/hmm/classifier_vs_hmm.csv` | C | TBD |
| HMM warm transition latency | 5.17 s for 19,313 obs (0.268 ms/transition); 4.62 s for 16,031 obs (0.288 ms/transition), train | `results/hmm/transition_benchmark.csv` | C | verified (Phase 1, defaults) |
| Integration HMM vs classifier (preliminary) | Top-1 0.851 vs 0.897; continuity 0.839 vs 0.756; LCS F1 0.729 vs 0.763 | `results/integration/per_trip_comparison.csv` | D → C | measured; C to tune |
| API p50 / p95 latency | HMM 1213 / 3579 ms; classifier 438 / 1179 ms | `results/integration/stage_timings.csv` | D | verified (preliminary artifact) |
| Student-test trips passing | 10/10 both modes; worst live API 4.07 s on 3,958 points | `results/integration/run_log.csv` | D | verified |
| Model artifact version | `6c229c610200` (untuned; refresh after Optuna) | `artifacts/model_card.json` | D | verified (preliminary) |

---

## 10. Q&A bank

### 10.1 Rapid-fire one-liners

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

### 10.2 Questions that get asked

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
| Did the HMM help? | It raises continuity but currently costs Top-1 with default parameters — decoder tuning is Phase 2 (see ledger). |
| How is ground truth kept server-side? | Prediction loads trips without `.route`; scoring runs afterwards and only metrics leave the API. |
| How do you prove the API is live? | The per-request log line with trip, latency and model version, plus the UI screenshot. |

### 10.3 Who must be able to explain what

| Member | Must know without notes |
|---|---|
| A | File semantics, inventory, projection, candidates, label alignment, flagged trips, maps |
| B | Features and ablation, CV protocol, metrics, model selection, Optuna |
| C | HMM states/emissions/transitions, Viterbi, gaps, decoder tuning and comparison |
| D | API contract, ground-truth isolation, two-laptop deployment, logging evidence, demo |

Two honest failure cases to have ready: a parallel-road confusion and a large GPS
gap (before/after map or metric).

---

## 11. Demo script (5–7 minutes)

1. **Context (30 s):** noisy GPS, recover the road sequence, UI never sees truth.
2. **UI (30 s):** health badge, trip list, pick a trip.
3. **Run (30 s):** click Run; point to the spinner and the API log line.
4. **Results (60 s):** metric cards, predicted route on the map, observed GPS.
5. **Pipeline (90 s):** candidates → features → classifier → HMM → Viterbi.
6. **Evidence (60 s):** model comparison, tuning table, classifier-vs-HMM, one failure.
7. **Honesty (30 s):** miss rate, flagged routes, what we would improve.

Show the system first, not slides.

---

## 12. Final checklist (T-60 minutes)

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

---

## 13. AI-assistance disclosure (draft)

> AI tools were used for repository scaffolding, code-review suggestions,
> debugging and language editing. All training, tuning, decoding, measurements
> and visualisations were executed by the team. Every member can explain the
> submitted code, results and decisions. No reported metric was generated
> without an artifact in the repository.

---

## 14. Keeping these notes current

This file is short on purpose, so it is cheap to keep true. Four rules:

1. **Numbers.** When a number changes, update its row in the ledger (§9) and any
   sentence that quotes it. Quote from the ledger, never keep a second copy.
2. **Behaviour.** If the pipeline, features, model choice, decoder or API contract
   changes, edit the one paragraph that describes it. Keep it short.
3. **Questions.** When someone is asked something we could not answer, add it to
   §10 with a one-line answer. That is how the bank grows.
4. **Ownership.** Whoever changes the behaviour updates this file in the same
   commit. A compiles it before a mock viva; D checks the demo evidence.

Audit before a milestone: `grep -n TBD viva.md` — every remaining `TBD` must be
planned Phase 2 work, not a forgotten number.
