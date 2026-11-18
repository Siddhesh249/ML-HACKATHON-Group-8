# PathSnap — Viva & Demo Preparation (Group 8)

> **Purpose:** the single internal document for the viva. It explains the system, the maths, the design decisions, the evidence and the demo. Every member must be able to explain any section, not only their own.
>
> **Rule:** no number is written here unless it exists in `results/`. Until measured, use `TBD`. Every number has a source path in the ledger in §18.
>
> **Status:** Phase 1 — A's data/EDA/label/candidate numbers are measured (T10–T12). Model/HMM/API numbers are `TBD` until their owners fill them.
**Last updated:** 2026-11-18 · **Maintainers:** all members; A compiles; D checks the demo evidence.

> **Agents:** read [`AGENTS.md`](AGENTS.md) and [`docs/STATUS.md`](docs/STATUS.md) before changing anything. Record changes in [`docs/AGENT_LOG.md`](docs/AGENT_LOG.md); this document is the viva-facing explanation, not the session log.

---

## 0. FIRST VIVA — task, dataset, preprocessing and features (read this first)

The first viva is expected to focus on **understanding the task**, the **dataset**, **preprocessing**, and **feature-engineering decisions** — before model results are final. This section is the one-page survival pack. The rest of the document is the deeper reference.

### 0.1 The task in 90 seconds

> We are given a noisy GPS trace from a vehicle and the local road network around that trace. The trace is sampled at roughly 1 Hz and the GPS is wrong by several metres; near buildings it can jump onto a parallel road or leave a gap. The road network is a directed graph: nodes are coordinates, arcs are road segments, and every arc has an ID (its row index).
>
> The job of PathSnap is **map matching**: for every GPS observation, decide which road arc the vehicle was actually on, then reconstruct a route that is connected in the road network. We do this in two stages. First, candidate generation: for each observation we shortlist nearby arcs and compute geometric/topological features. A classifier scores each candidate and gives a probability. Second, an HMM with Viterbi decoding uses the road-network connectivity and the time between observations to choose a coherent sequence, instead of independent per-point guesses.
>
> We train on 80 trips, compare at least five model families using group-aware cross-validation, tune the selected model with Bayesian optimisation, and evaluate classifier-only versus classifier-plus-HMM on 10 student-test trips. The ten evaluator trips are reserved for the final live demonstration, where a FastAPI backend on one laptop serves a UI on a second laptop. Ground truth stays on the server and is never shown to the UI.

Then stop. The next question is usually “show me the data”.

### 0.2 Dataset one-pager (be able to draw this)

```
trip folder: data/train/00000001/
    00000001.track    lon, lat, timestamp_s      (N rows, noisy GPS, ~1 Hz)
    00000001.nodes    lon, lat                   (M rows, node ID = row index)
    00000001.arcs     from_node, to_node         (A rows, arc ID = row index)
    00000001.route    arc_id                    (R rows, reference arc sequence)
```

| Question | Answer |
|---|---|
| What is a node / arc / route? | Node = road-network coordinate with a row-index ID; arc = directed segment between two nodes, ID = row index; route = chronological sequence of arc IDs the vehicle traversed. |
| Is one `.route` row one GPS point? | No. A route is an arc sequence; many GPS points project onto one arc, so the route usually has far fewer rows than the track. |
| Are node IDs global? | No. Each trip ships its own network, often in a different city; IDs are only valid inside that trip. |
| How many trips? | 80 train, 10 student-test, 10 evaluator (not in the student package). |
| What are the trip counts? | 80 train (208,417 GPS points), 10 student-test (19,160), 10 evaluator (brief). Per trip: 262–19,313 points, 299–155,003 nodes, 649–307,392 arcs, 59–2,266 route arcs (`results/eda/summary.json`). |
| Are there missing values? | Measured: 0 non-finite/out-of-range GPS rows and 0 repeated/backward timestamps in all 90 trips. The issues are structural instead: 106 zero-length arcs, 24 self-loops, 1,903 duplicate arcs, a median of 19 disconnected components per network, and gaps up to 153,064 s (`results/eda/quality_checks.csv`). |
| What are the main data-quality issues? | GPS noise/jitter, irregular sampling and gaps, contracted long arcs that cut across curves, and a few partial/offset reference routes. |
| Which trips have route problems? | Unreliable (coverage < 70%): train `00000054` (11.5%, steady 100–200 m offset for 4.5 h) and `00000033` (29.1%, vehicle parked ~120 m off-network). 11 routes have 1–3 breaks (15–597 m); 8 are loop/out-and-back. Student-test: none unreliable (`results/labels/coverage.csv`). |

**Why this matters:** the model must not memorise coordinates or node IDs. Every feature is relative to the current trip and candidate set, so it transfers to an unseen network.

### 0.3 Preprocessing pipeline in six steps

1. **Discover and validate** the trip: required files, column counts, lon/lat range, monotonic timestamps, arc endpoints in range, route IDs in range. A malformed trip is rejected, not silently loaded.
2. **Project to metres** with a local equirectangular projection around the trip’s mean latitude:
   $x=(\lambda-\lambda_0)111320\cos\phi_0$, $y=(\varphi-\varphi_0)110540$.
   Justify: angular degrees are not distances; a local tangent plane is accurate at trip scale and keeps vector maths fast.
3. **Smooth the trace** over a short centred window (default 5 observations), then compute heading and speed from the smoothed positions and the real timestamps. Raw 1 Hz bearings and speeds are dominated by jitter.
4. **Generate candidates** for each observation: a KD-tree over arcs sampled every ~5 m, radius 60 m (expand to 100/150 m if empty), exact point-to-segment distance, keep the top 15. Sample along the arc, not just endpoints, because long arcs are straight chords across curved roads.
5. **Align the reference route** to the track: build the route polyline, project each GPS point onto it, enforce monotonic progress, map back to an arc. Flag observations whose distance to the assigned segment exceeds 30 m as low confidence.
6. **Build features** per (observation, candidate) pair, fit scalers inside the training folds only, and never use evaluator data.

### 0.4 Feature-selection rationale (be ready to defend each group)

| Group | Features | Why it helps | Alternative considered / rejected |
|---|---|---|---|
| Geometric | perpendicular distance, distance ratio to best candidate, projection ratio, distance to arc endpoints | The closest road is usually right, but not at junctions or parallel roads; the ratio and projection position add context | Raw lon/lat: rejected, it memorises the city and does not transfer |
| Directional | absolute heading difference, cosine of heading difference | A vehicle cannot travel sideways along a road; heading separates parallel/opposite carriageways | Raw consecutive bearing: rejected, too noisy at 1 Hz; we use a smoothed heading |
| Topological | arc length, node in/out degree, reciprocal-arc flag, graph distance from the previous best candidate, same-arc flag, graph-distance vs GPS-displacement residual | Encodes connectivity and junction complexity, which distance alone cannot; the residual catches teleports | Only distance: rejected; it produces impossible jumps |
| Temporal | smoothed speed, log gap seconds, candidate count | Speed rejects candidates that imply impossible movement; the gap tells the model and decoder when to relax; candidate count calibrates the ranking | Fixed thresholds only: rejected; gaps vary from 1 s to hours |
| Labels | reference arc from route alignment, confidence flag | Supervised target without treating the route as infallible truth | Nearest arc to the GPS point as a label: rejected, it reproduces GPS noise |

**Leakage discipline:** no target-derived feature, no evaluator data, no scaler fitted on validation/test rows, no trip split across folds, no raw coordinates or IDs.

### 0.5 Whiteboard sketches you must be able to draw

1. **Coordinate projection:** a small tangent plane with $x$ east, $y$ north, and the metre scaling.
2. **Point-to-segment distance:** point $P$, segment $AB$, projection $P^*$, ratio $t\in[0,1]$.
3. **Candidate pipeline:** GPS point → KD-tree radius query → unique arcs → exact distance → top-K.
4. **Label alignment:** route polyline, GPS points projected onto it, monotonic arclength, arc under each projection.
5. **Feature vector:** candidate row with the 18 columns and a one-line reason for each group.
6. **Two-stage model:** candidate classifier probabilities → HMM emissions; road graph + time → transitions; Viterbi → route.

### 0.6 First-viva rapid questions (25)

| # | Question | Answer in one or two sentences |
|---|---|---|
| 1 | What is map matching? | Assigning noisy GPS observations to road-network arcs and reconstructing a connected route. |
| 2 | What are the inputs and outputs? | Inputs: track, nodes, arcs (and route for supervision/scoring). Output: one predicted arc per observation and a route arc sequence. |
| 3 | Why is the route not a GPS label file? | It lists traversed arcs chronologically; multiple observations map to the same arc, so it must be aligned before it can supervise a per-point classifier. |
| 4 | Why do trips have different networks? | The dataset ships a local subgraph per trip; there is no shared global graph. |
| 5 | What is an arc ID? | The 0-based row index in the `.arcs` file. |
| 6 | What is a node ID? | The 0-based row index in the `.nodes` file. |
| 7 | Why not use Euclidean distance in degrees? | Degrees are angular; longitude shrink depends on latitude. We project to metres or use geodesic distance. |
| 8 | What projection do you use? | Local equirectangular around the trip’s mean latitude; error is negligible at trip scale. |
| 9 | Why smooth the GPS? | 1 Hz jitter makes raw heading and speed unreliable; a centred window reduces noise without using labels. |
| 10 | Does smoothing leak future information? | It uses neighbouring observations in the same trip only, applied identically at training and inference; it never uses the target. |
| 11 | Why candidates instead of all arcs? | The network can have hundreds of thousands of arcs; a shortlist makes training and inference tractable and lets us measure recall. |
| 12 | Why sample along arcs for the index? | Arcs are contracted chords; endpoints alone can be far from a point on the road. |
| 13 | What is the candidate miss rate? | The fraction of labeled observations whose true arc is absent from the candidate set; it is a hard ceiling on Top-1 accuracy. A reports the measured value from `results/eda/candidate_sweep.csv`; do not quote a number before it exists. |
| 14 | How do you handle no candidates? | Expand the radius; if still empty, mark the observation unmatched and report it. |
| 15 | How are labels created? | Project the track onto the route polyline, enforce monotonic progress, map the arclength to an arc, and flag low-confidence points. |
| 16 | Why enforce monotonic progress? | The vehicle cannot teleport backwards along the route; this removes projection ambiguity. |
| 17 | What do you do with low-confidence labels? | Exclude them from training and report coverage; do not teach the classifier a wrong arc. |
| 18 | Which features are most important and why? | Fill from the actual importance plot. Expected: distance, heading agreement, graph continuity; they separate nearby alternatives and prevent teleports. |
| 19 | Why no arc/node ID features? | They are trip-local; the model would memorise the training cities and fail on an unseen network. |
| 20 | Why log-transform distances and lengths? | They are heavy-tailed; logs stabilise linear models and neural networks and compress outliers. |
| 21 | What is the biggest data-quality risk? | Bad reference-route alignment, because it corrupts labels and evaluation; hence confidence flags and flagged-trip reporting. |
| 22 | How do you avoid evaluator leakage? | The prediction path never opens `.route`; scoring is separate and runs after prediction; no evaluator data in training or tuning. |
| 23 | What is the evaluation unit? | Observation-level Top-1 for map matching; candidate-level precision/recall/F1/log-loss; route continuity and sequence LCS F1 for the decoder. |
| 24 | What is the first model you will compare? | A nearest-road baseline, then logistic regression; both are interpretable and expose how much the learned features add. |
| 25 | What is still `TBD` in this viva? | Model, Optuna, HMM and latency numbers. Say they are Phase 2 and point to the protocol; never invent a number. |

### 0.7 First-viva traps (do not say these)

- ❌ “Each row of the track corresponds to a row of the route.” → No, align them.
- ❌ “The route is exact ground truth.” → It is reference information with documented limitations.
- ❌ “We use latitude/longitude as features.” → We use trip-relative geometry only.
- ❌ “We snap each point to the nearest road.” → That is the baseline; it fails on parallel roads and at junctions.
- ❌ “We share one network across trips.” → Each trip has its own network.
- ❌ “Accuracy is 99%” before it is in `results/`. → Quote only measured values with a source.

### 0.8 First-viva demo (3–4 minutes, no model needed)

1. Open a trip folder and show the four files; explain the row-index IDs.
2. Show `scripts/01_eda.py` output: inventory, gap histogram, GPS/network map.
3. Show one observation’s candidate set: distances, headings, top-K.
4. Show the route alignment overlay: GPS points, reference polyline, confident vs flagged points.
5. Show the feature table header and explain two rows.
6. Close with the two-stage model plan and what Phase 2 will measure.

If the model is not ready, do **not** improvise a prediction. Show the pipeline and the plan; the first viva is about understanding.

---

## 1. How to use this document

| When | Who | What |
|---|---|---|
| Start of each stage | all | Read the sections you own and the section you will be quizzed on |
| After every experiment | experiment owner | Update the numbers tables + ledger in the **same PR** |
| After M2 (Phase 1) | A + C | Mock viva round 1: each person answers questions on a section they do **not** own |
| After M4 (integration) | A + C | Mock viva round 2 + full demo rehearsal |
| T-60 min | all | Run the final checklist (§22), read §23, print the demo script |

**Golden rules**

1. Explain *why*, then *how*, then the number. “We chose it because…” beats a definition.
2. Never claim a result that is not in `results/`.
3. If you do not know, say what you would measure next. Do not improvise a metric.
4. Know one failure case of your own component. Evaluators respect honest error analysis.
5. Keep the API terminal visible during the demo: it is the proof that inference is live.

---

## 2. Two-minute pitch (memorise the flow, not the words)

> GPS is noisy, especially in cities. A vehicle can appear on a parallel street, jump across buildings, or leave a gap. PathSnap reconstructs the true route by matching GPS observations to a road network.
>
> We treat map matching as a two-stage problem. First, for each GPS point we generate the nearby road segments as candidates and train a classifier to score each candidate using geometry and topology: perpendicular distance, projection position, heading agreement, road-arc length, node degree and graph reachability from the previous point. The classifier gives us calibrated candidate probabilities.
>
> Second, independent predictions can jump between disconnected roads. We therefore run an HMM over the candidate arcs. The hidden state is the road segment at each observation; the emission comes from the classifier; the transition comes from the road network, comparing the network shortest-path distance with the GPS displacement. Viterbi finds the most probable connected sequence in log space, and we handle GPS gaps by resetting the chain.
>
> The final system runs the model and ground-truth scoring on Laptop 1 behind a FastAPI service, and a thin Streamlit UI on Laptop 2. The UI never receives ground truth. On the reserved trips, the evaluator selects a trip, the API runs the full pipeline, and the score is computed server-side from reference data that the predictor never sees.
>
> We trained and compared at least five model families, selected the final model on group-aware cross-validation over the 80 training trips, tuned it with Bayesian optimisation, and compared classifier-only predictions against the HMM decoder on pointwise accuracy, route continuity and sequence-level F1.

Rehearse this to **under 2 minutes**. Then stop and invite questions.

---

## 3. System overview

```
             ┌──────────────────────────── Laptop 1 ────────────────────────────┐
             │                                                                 │
 .track ───► │  loader/validator ─► local metric projection ─► candidates       │
 .nodes ───► │       │                                              │            │
 .arcs  ───► │       │                                              ▼            │
             │       │                                   feature matrix (18)     │
             │       │                                              │            │
             │       │                                              ▼            │
             │       │                                   calibrated classifier    │
             │       │                                              │            │
             │       │                                              ▼            │
             │       │                              HMM + Viterbi decoder          │
             │       │                                              │            │
 .route ───► │  scoring (separate module, after prediction) ◄────────┘            │
 (server     │       │                                                          │
  only)      │       ▼                                                          │
             │   FastAPI  /health /version /trips /predict                       │
             └───────────────────────────┬─────────────────────────────────────┘
                                         │ HTTP over LAN
             ┌───────────────────────────▼─────────────────────────────────────┐
             │                    Laptop 2 — Streamlit UI                      │
             │  trip list · run prediction · metric cards · map · timings      │
             │  (never receives reference route or ground-truth labels)        │
             └─────────────────────────────────────────────────────────────────┘
```

**One-sentence architecture:** a two-tier distributed system where all learning, decoding and scoring happen server-side, and the client is a thin presentation layer.

Why “distributed”? The brief specifies a two-laptop deployment: Laptop 1 hosts the model/API/HMM/road network; Laptop 2 hosts the evaluator UI. It is a deployment distribution, not distributed training.

---

## 4. Dataset: what each file means

| File | Columns | Meaning | Key subtlety |
|---|---|---|---|
| `.track` | `lon`, `lat`, `timestamp_s` (3 cols, no header) | Noisy GPS observations of one trip, nominally 1 Hz | Not every step is 1 s; gaps exist; coordinates are longitude first |
| `.nodes` | `lon`, `lat` (2 cols, no header) | Road-network nodes; the **0-based row index is the node ID** | Node IDs are local to the trip |
| `.arcs` | `from_node`, `to_node` (2 cols, no header) | Directed road segments; the **0-based row index is the arc ID** | Both directions of a road usually appear as separate arcs |
| `.route` | `arc_id` (1 col, no header) | Chronological reference arc sequence traversed by the vehicle | A route, not a per-GPS-point label; it has fewer rows than the track |

**Partitions**

| Partition | Trips | Use |
|---|---|---|
| Training | 80 | Fitting, cross-validation, model selection, Bayesian optimisation |
| Student testing | 10 | Development-stage comparison; decoder comparison |
| Evaluator testing | 10 | Final evaluation only; never used for training, tuning, thresholds or feature choice |

**Leakage rules**

- No trip may appear in more than one partition; splits are grouped by trip.
- The evaluator partition is never used for hyperparameters, thresholds, model choice or feature selection.
- Preprocessing statistics (scalers) are fit on training data only.
- `.route` is not read by the prediction path; it is opened only by the scoring module after prediction.
- No hardcoded outputs and no per-trip overrides.

**Verified inventory** (re-run by T10; see `docs/DATA_NOTES.md`)

**Inventory to measure (A, T10)** — do not quote counts until the evidence exists.

| Fact | Value |
|---|---|
| Trips provided | 80 train + 10 student-test |
| Total size | A to measure |
| Train GPS observations | A to measure |
| Student-test GPS observations | A to measure |
| Nodes per trip | A to measure |
| Arcs per trip | A to measure |
| Reference arcs per trip | A to measure |
| Median GPS-to-road offset | A to measure |
| Trips with large gaps | A to measure |
| Largest time gap | A to measure |

**Three dataset facts that shape the design**

1. **Networks are per-trip and city-specific.** Two trips may share no nodes. A model that memorises coordinates or node IDs will not generalise. Features must be relative: distances, ranks, angles, degrees, reachability.
2. **Reference routes are useful but imperfect.** Some are partial or offset (e.g. train `00000033`, `00000054`; student-test `00000077`). We treat the route as reference information, not infallible truth: we align it, score our confidence, and report the affected trips.
3. **GPS noise is substantial at 1 Hz.** Consecutive positions jitter by several metres, so raw speed and bearing are unreliable. We smooth positions and headings over a short window before computing features.

---

## 5. Exploratory data analysis (what we looked at and why)

Required outputs (see `scripts/01_eda.py`):

1. Dataset inventory: trips, files, observations, nodes, arcs, route lengths.
2. Summary statistics for trajectory length, sampling intervals, missing/invalid values.
3. GPS coordinate distributions and trajectory plots.
4. Interactive Folium maps of selected trajectories + road network.
5. Road-network size and connectivity tables/visualisations.
6. A data-quality record: assumptions, anomalies, decisions.

Key analyses to be able to explain:

- **Sampling interval histogram:** 1 Hz nominal, but gaps. We measure the tail and choose a gap threshold (candidate: 5 s for features, 30 s for HMM resets — confirm with tuning).
- **Arc-length distribution:** median is tens of metres, but some arcs are kilometres long. They are contracted chords: the road curves, the arc is a straight line between endpoints. This affects distance features and candidate indexing.
- **Connectivity:** per-trip networks have multiple connected components (undirected). A predicted sequence that jumps components is impossible; the HMM transition must forbid it.
- **Missing/invalid:** no NaNs in the provided files, but zero-length arcs, isolated nodes and coordinate outliers exist. The loader validates shapes and indices; the pipeline guards divisions by zero.
- **GPS-to-route distance:** a histogram showing most points within a few metres, with a tail caused by noise, route chords and (in a few trips) route misalignment. This motivates confidence flags.
- **Candidate miss rate vs radius:** the share of labeled observations whose reference arc is absent from the candidate set. It is the ceiling on Top-1 accuracy and must be reported.

**Notebook vs script:** EDA lives in `scripts/01_eda.py` so figures regenerate from data. Any notebook is exploratory and must not contain the only copy of a result.

---

## 6. Preprocessing and geographic calculations

### 6.1 Coordinate handling

- Coordinates are `(longitude, latitude)` in degrees.
- Euclidean distance in degrees is not a physical distance. At latitude $\phi$, one degree of longitude is $111{,}320\cos\phi$ m, one degree of latitude is about $110{,}540$ m.
- We use a **local equirectangular (tangent-plane) projection per trip** around a reference latitude:

$$x = (\lambda-\lambda_0)\,111320\cos\phi_0,\qquad y = (\varphi-\varphi_0)\,110540$$

- At the scale of a trip (a few tens of km) the error relative to geodesic distance is well under 0.5%; it keeps vectorised geometry fast. Distances are reported in metres.
- Alternative considered: UTM or pyproj. Rejected for the hackathon because the trips span different UTM zones and the local plane is simpler; the error is negligible at this scale.

### 6.2 Point-to-segment distance

For a GPS point $P$ and an arc segment $AB$:

$$\mathbf{t}=\operatorname{clip}\!\left(\frac{(P-A)\cdot(B-A)}{\lVert B-A\rVert^2},0,1\right),\qquad P^\* = A + t(B-A),\qquad d=\lVert P-P^\*\rVert$$

- $t$ is the **projection ratio**; $t=0$ or $1$ means the closest point is an endpoint.
- $d$ is the perpendicular (or endpoint) distance used in features and labels.
- Guard: if $\lVert B-A\rVert^2\approx 0$ (zero-length arc), use the endpoint distance.

### 6.3 Heading and speed

- Bearing from point $i$ to $j$:
  $$\theta=\operatorname{atan2}\!\big((\lambda_j-\lambda_i)\cos\bar\varphi,\ \varphi_j-\varphi_i\big)$$
- Heading difference is wrapped to $[-180^\circ,180^\circ]$; features use $|\Delta\theta|$ and $\cos\Delta\theta$.
- Speed is computed from **smoothed** positions and the actual timestamp difference, never from raw consecutive points.
- Smoothing: centred moving average over a short window (e.g. 5 points) with edge padding. Window length is a tunable preprocessing choice, fit on training data only.

### 6.4 Why we do not use raw `lat`/`lon` as features

They encode the city. A tree could split on a coordinate and memorise one city; the model would fail on a new network. We use coordinates only to compute relative geometry within the current trip.

---

## 7. Candidate generation

**Purpose:** reduce “which of hundreds of thousands of arcs?” to “which of ~15 nearby arcs?”.

**Algorithm**

1. Project the trip into local metres.
2. Build a spatial index over **dense samples of each arc** (e.g. every 5 m, capped per arc). Sampling endpoints only is insufficient because arcs are long chords that cut across curves.
3. Query a KD-tree with radius $r$ (default 60 m) for every GPS point.
4. Deduplicate sampled arc IDs, compute exact point-to-segment distance for those arcs, keep the top-$K$ (default 15).
5. If the candidate set is empty, expand the radius (e.g. 100 m, then 150 m) and retry. Record the fallback.
6. Optional refinement: include arcs reachable from the previous candidate within the maximum plausible travel distance. This improves recall at junctions but increases candidate count.

**Why KD-tree over arc samples:** exact segment distance against all arcs is $O(N\cdot M)$; for a 300k-arc network that is prohibitive per trip. Sampling makes the index dense enough that any nearby arc has a sampled point within a few metres, then exact geometry is computed only for the shortlist.

**Candidate miss rate** = fraction of **labeled** observations whose reference arc is not in the candidate set. It is a hard ceiling: no classifier or decoder can recover a missing truth. We report it overall, per trip, and as a function of radius. Observations with a missing truth are excluded from training labels (not silently labeled negative).

**Viva traps**

- “Why not simply snap to the nearest arc?” → that is our baseline; noise and parallel roads make it wrong at intersections and on divided highways. We compare against it.
- “Why 60 m?” → measured, but answer honestly: on labelled points the radius hardly matters (K=15 miss ≈ 0.09% for every radius from 30 to 100 m) because a label is only confident when the point is ≤ 30 m from the route, so any radius ≥ 30 m finds it almost by construction. The sweep therefore cannot justify a *small* radius. We keep 60 m + 100/150 m fallbacks for the unlabelled points the model meets at inference (GPS-to-route p90 ≈ 48 m), which a 30 m radius without fallback would leave with no candidates.
- “Why top-K 15?” → top-K is what drives the miss rate: K=5 ≈ 1.2%, K=10 ≈ 0.24%, **K=15 ≈ 0.09%**, K=20 ≈ 0.035%, K=30 ≈ 0.002% (train, no fallback). K=15 is the knee: going to 20 buys 0.06 pp for +13% candidates. Production setting: 0.092% train / 0.032% student-test, 8.7 candidates per observation, never empty (`results/eda/candidate_sweep.csv`, `candidate_sweep_choice.json`, figure `candidate_sweep.png`). Chosen on train only.
- “What if a long arc has no sample near the point?” → samples are spaced at 5 m and capped only for very long arcs; the cap is documented and the exact distance check removes false candidates.

---

## 8. Labels from the reference route

The `.route` file is a sequence of arcs, not one label per GPS point. To train a per-observation classifier we must align the track to that sequence.

**Procedure**

1. Build the reference polyline by concatenating the route arcs in order (each arc is a straight segment between its endpoint nodes).
2. Assign each segment its arc ID and cumulative arclength $s$.
3. For each GPS point, find the nearest segment and its arclength coordinate.
4. Enforce **monotonic progress** along the route: the vehicle cannot teleport backwards. Implementation: monotone projection (running maximum of $s$), with a dynamic-programming variant kept for routes that revisit an arc.
5. Map the monotonic arclength back to the arc ID.
6. Record the distance from the point to the assigned segment. If it exceeds a threshold (e.g. 30 m), mark the label **low confidence**.
7. Per trip, report: labeled fraction, median/p90 alignment distance, fraction of backward moves before monotonic enforcement, and whether the route covers the whole trip.

**Why this matters**

- Training on a misaligned route teaches the classifier the wrong arc. Confidence flags prevent that.
- Evaluation must use the same alignment for every model. Otherwise “Top-1 accuracy” is not comparable.
- Some routes are genuinely partial. We report them rather than hiding them; this is an honest data limitation, not a modelling failure.

**Limitations to state in the viva**

- The reference route is not independently verified physical truth.
- A route may revisit an arc (loops); nearest-segment alignment can be ambiguous. The monotonic/DP alignment resolves the common case; ambiguous cases are flagged.
- Projection distance includes both GPS noise and chord approximation of curved roads.

---

## 9. Features

All features are computed **per (observation, candidate arc) pair** and are trip-relative. Proposed set (frozen in `pathsnap/config.py`):

| # | Feature | Intuition |
|---|---|---|
| 1 | `log1p(distance_m)` | Perpendicular/endpoint distance; compresses the long tail |
| 2 | `distance_ratio_to_best` | Distance divided by the closest candidate's distance; relative, scale-free |
| 3 | `candidate_rank` | 0 = closest; captures the ordering within the shortlist |
| 4 | `projection_ratio` | Where along the arc the point projects; distinguishes mid-arc from junction |
| 5 | `abs_heading_diff_deg` | Agreement between smoothed travel direction and arc direction |
| 6 | `cos_heading_diff` | Smooth, bounded version of the same signal |
| 7 | `log1p(arc_length_m)` | Long chords behave differently from short junction arcs |
| 8 | `log1p(dist_to_arc_start_m)` | Proximity to the arc's entry node |
| 9 | `log1p(dist_to_arc_end_m)` | Proximity to the arc's exit node |
| 10 | `node_degree_from` | Junction complexity at the entry node |
| 11 | `node_degree_to` | Junction complexity at the exit node |
| 12 | `has_reciprocal_arc` | Divided roads / two-way roads have a reverse arc; useful context |
| 13 | `log1p(graph_dist_from_prev_best_m)` | Network reachability from the previous best candidate |
| 14 | `same_arc_as_prev_best` | Vehicle may stay on the same road for several observations |
| 15 | `log1p(abs(graph_dist_minus_gps_disp_m))` | Consistency between network travel and GPS displacement |
| 16 | `log1p(speed_mps)` | Smoothed speed; helps reject candidates that imply teleportation |
| 17 | `log1p(gap_seconds)` | Time since previous observation; large gaps relax transition expectations |
| 18 | `candidate_count` | Candidate-set size for this observation; context for ranking |

**Rules**

- No raw coordinates, node IDs, arc IDs, trip IDs or partition flags.
- No target-derived features.
- Scalers are fit on training folds only (inside the pipeline, not before splitting).
- Feature groups are ablated: local only → + relative → + temporal/graph. Report CV Top-1 for each group.

**Viva questions**

- “Why is perpendicular distance not enough?” → parallel roads, GPS offset and long chords; distance alone confuses nearby alternatives.
- “Why include graph distance to the previous candidate?” → it encodes continuity, which is exactly what map matching needs. It is still a per-candidate feature; the full sequence optimisation happens in the HMM.
- “Why log-transform?” → heavy-tailed distances and lengths; trees are invariant but linear models and neural nets benefit.
- “Why no `arc_id` one-hot?” → arc IDs are trip-local and would not generalise; it would memorise the training networks.

---

## 10. Models

At least five distinct families are trained and compared (target: 7).

| Model | Family | Why included | Main limitation |
|---|---|---|---|
| Nearest-road baseline | Rule-based | Establishes the value of learning; fast | Ignores sequence and heading |
| Logistic regression | Linear / probabilistic | Interpretable, calibrated, strong baseline | Linear decision boundary |
| Gaussian Naive Bayes | Probabilistic | Very fast; shows the effect of feature independence assumption | Strong independence assumption; poor calibration on correlated features |
| Decision tree | Tree | Nonlinear, interpretable splits; exposes overfitting | High variance; needs depth/leaf limits |
| Random forest / Extra Trees | Ensemble | Stable variance reduction; feature importance | Larger model; slower inference |
| HistGradientBoosting | Boosting | Strong tabular performance; fast; handles nonlinearity | Less interpretable; early stopping needed |
| Shallow MLP | Neural network | Learns smooth nonlinear interactions | Needs scaling, seeds and early stopping; least stable |
| Linear SVM (calibrated) | Kernel/margin | Maximum-margin alternative; calibrated for probabilities | Calibration overhead; sensitive to scaling |

**Training protocol**

- Binary target: candidate is the reference arc (label from §8), one row per candidate.
- Class imbalance is severe (~1 positive per 10–40 candidates). Two mitigations: `class_weight` and negative subsampling (all positives + 2–3 negatives per positive). Evaluate on the **full** candidate set.
- GroupKFold(5) by trip on the 80 training trips. Never random-split observations from the same trip across folds.
- Standardise inside a pipeline where the model needs it.
- Fix and record seeds; log package versions and the git commit.
- Record training time and per-trip inference time.
- For probability models, use `predict_proba`; for margin models, calibrate (Platt/isotonic) before computing log loss and HMM emissions.

**Per-model questions the report must answer**

1. Why is this algorithm suitable for PathSnap?
2. Strengths and limitations for this task.
3. Important hyperparameters and initial values.
4. Loss/objective used.
5. Student-test performance.
6. Observation Top-1, macro F1, log loss (where applicable).
7. Evidence of over/underfitting.

**Overfitting evidence:** compare CV train vs validation metrics, fold variance, and learning curves only where the model produces real epoch-wise histories (MLP, HistGB). Never draw fabricated training curves for models without histories.

---

## 11. Evaluation protocol and metric definitions

| Metric | Definition | Why |
|---|---|---|
| Candidate precision | TP / (TP + FP) over candidate pairs | How often a high-confidence candidate is wrong |
| Candidate recall | TP / (TP + FN) | How many reference arcs are retrieved and scored correctly |
| Positive F1 | Harmonic mean of the two | Balanced view under imbalance |
| Macro F1 | Mean of positive and negative F1 | Required by the brief; shows both classes |
| Log loss | $-\frac1N\sum[y\log p+(1-y)\log(1-p)]$ | Probability quality; matters for HMM emissions |
| Observation Top-1 | Argmax candidate per observation equals the aligned reference arc | Direct map-matching quality, fixed candidate sets |
| Candidate miss rate | Labeled observations whose reference arc is not a candidate | Ceiling on Top-1; data/pipeline diagnostic |
| Route continuity | Fraction of consecutive predicted arcs that are graph-connected | Penalises impossible jumps |
| Sequence LCS F1 | Precision/recall/F1 from the longest common subsequence of predicted and reference arc sequences | Order-aware route quality |
| Exact sequence match | Predicted sequence equals reference | Strict, usually low; useful diagnostic |
| Inference time | Wall clock per trip and per stage | Deployment feasibility |

**Primary selection metric:** CV **observation Top-1** on fixed candidate sets, because it directly measures the map-matching decision and is comparable across models that see identical candidates.
**Guardrails:** positive-class F1 (imbalance), log loss (probability quality for the HMM), fold stability (std across folds).
**Final gate:** HMM route continuity and sequence F1 on student-test.
**Why not accuracy?** On candidate pairs, accuracy is dominated by negatives (~90%); a model that always predicts “not this arc” scores highly. Macro F1 and Top-1 expose that.

**Protocol**

- 80 training trips → GroupKFold(5) for every model.
- 10 student-test trips → final comparison of the selected/untuned/tuned models and the decoder; never used to tune.
- Evaluator partition → untouched until the final evaluation.
- Same candidate sets, labels, and metrics for every model.

**Tuned vs untuned comparison** uses the same folds, same features, same metrics, and records the computational cost of the search.

---

## 12. Bayesian hyperparameter optimisation

**Why Bayesian?** Each trial costs a full grouped-CV fit. Grid search scales exponentially; random search ignores what it learns. A Bayesian optimiser models the objective and spends trials where improvement is likely.

**Tool:** Optuna with the TPE sampler (Tree-structured Parzen Estimator). Fallback: scikit-optimize GP or a documented Bayesian method if Optuna is unavailable.

**Study design**

- Objective: mean CV Top-1 across 5 GroupKFold folds (tie-break: positive F1, then log loss).
- Search space (example for HistGradientBoosting; adjust for the selected model):
  - `learning_rate`: log-uniform 0.01–0.3
  - `max_leaf_nodes`: 15–127
  - `min_samples_leaf`: 20–200
  - `l2_regularization`: log-uniform 1e-8–10
  - `max_iter`: 100–600 (with early stopping)
- Trials: ≥30, time-boxed; record every trial, including failures.
- Group-aware CV: folds are trips, never observations.
- Refit the best configuration on all 80 training trips.
- Evaluate on the 10 student-test trips and compare against the untuned model.
- Record wall-clock cost of the whole study.

**Never tune on evaluator data.** The search space, objective, trial count and threshold choices are fixed before seeing evaluator results.

**Artifacts**

- `results/optuna/study.db` (or journal/JSON), `trials.csv`, best params JSON.
- Optimisation history, parameter-importance and parallel-coordinate plots.
- A short note on what the optimiser found and whether it was worth the cost.

**Viva traps**

- “Why is this Bayesian and not just random search?” → TPE models $p(\text{params}\mid\text{good score})$ and $p(\text{params}\mid\text{bad score})$ and proposes candidates maximising the ratio; random search has no memory.
- “Did it actually improve anything?” → report the measured delta, even if zero/negative. Negative results are valid.
- “Why not tune the threshold too?” → thresholds are part of the decision rule; if tuned, they must be tuned inside CV on training trips only, then frozen.

---

## 13. HMM and Viterbi

### 13.1 Why an HMM

The classifier scores candidates independently. Consecutive observations can be assigned to arcs that are not connected, producing teleports. The HMM adds two things: **sequence context** and **road-network feasibility**.

### 13.2 Hidden states

The hidden state at observation $t$ is the **candidate arc** assigned to that observation. The state space is the candidate set $C_t$ (not all arcs), which keeps the correspondence between candidate, arc and observation and makes the decoder tractable.

### 13.3 Emission

For candidate $j$ at observation $t$:

$$e_t(j)=p_\text{model}(y=1\mid x_{tj})$$

Normalise over the candidate set (they are not mutually exclusive probabilities):

$$\tilde e_t(j)=\frac{e_t(j)}{\sum_{k\in C_t} e_t(k)}$$

Optionally blend with a geometric likelihood:

$$\log \tilde e_t(j) \leftarrow \log \tilde e_t(j) - \lambda\,\frac{d_t(j)^2}{2\sigma^2}$$

- $d_t(j)$ is the point-to-arc distance.
- $\lambda$ and $\sigma$ are decoder hyperparameters tuned on permitted data only.
- If all model scores are zero, fall back to a uniform emission.

### 13.4 Transition

For candidate $i$ at $t-1$ and candidate $j$ at $t$:

1. Compute the network travel distance from $i$'s projection to $j$'s projection:
   $$d_\text{net}(i,j)=\big(\ell_i - s_i\big)+\operatorname{sp}(v_i,u_j)+s_j$$
   where $\ell_i$ is the arc length, $s_i$ the projection offset from the arc start, $\operatorname{sp}(v_i,u_j)$ the directed shortest-path distance from $i$'s exit node to $j$'s entry node, and $s_j$ the projection offset on $j$.
2. Estimate the observed displacement $d_\text{gps}$ between the two GPS points.
3. Reject infeasible transitions: if $d_\text{net}$ is infinite or exceeds $v_\text{max}\Delta t + \epsilon$, set the log-probability to $-\infty$.
4. Otherwise:
   $$\log a_{ij}=-\frac{\lvert d_\text{net}-d_\text{gps}\rvert}{\beta}-\gamma\,\mathbb{1}[\text{U-turn}]$$
5. Row-normalise $\log a_{ij}$ for numerical stability.

Why network distance and not Euclidean? A candidate pair may be close as the crow flies but separated by a river or a one-way system. Network distance reflects what the vehicle can actually do.

### 13.5 Viterbi recurrence

Let $\delta_t(j)$ be the best log-probability of any state sequence ending in candidate $j$ at time $t$.

Initialisation:

$$\delta_1(j)=\log \tilde e_1(j)+\log \pi_j$$

Recursion:

$$\delta_t(j)=\log \tilde e_t(j)+\max_{i\in C_{t-1}}\Big[\delta_{t-1}(i)+\log a_{ij}\Big]$$

Backpointer:

$$\psi_t(j)=\arg\max_{i\in C_{t-1}}\Big[\delta_{t-1}(i)+\log a_{ij}\Big]$$

Termination and backtracking:

$$j_T^\*=\arg\max_{j\in C_T}\delta_T(j),\qquad j_{t}^\*=\psi_{t+1}(j_{t+1}^\*)$$

All computation is in log space to avoid underflow. Complexity is $O(T\cdot K^2)$ pair evaluations plus the cost of the shortest-path queries.

### 13.6 Gaps, missing candidates and invalid transitions

| Case | Handling |
|---|---|
| No candidate within the maximum radius | Skip that observation’s state layer; carry the previous state forward; mark the output as “no match” |
| Large GPS gap ($\Delta t$ above threshold) | Reset the chain: allow any candidate with a weak gap-aware prior; or restart Viterbi at that point |
| Disconnected candidates | Transition probability $-\infty$; the decoder must choose a connected alternative or flag an impossible route |
| Candidate set changes drastically | Normalise emissions per observation; do not compare raw scores across different candidate-set sizes |
| Route revisits an arc | The state is the candidate occurrence, not the arc identity alone; keep the observation index in the state |
| Evaluator trip has a huge point count | Restrict shortest paths to a corridor subgraph; cache per candidate node; enforce a response-time budget |

### 13.7 Decoder experiments

Compare **classifier-only** vs **classifier + HMM**:

- Observation Top-1.
- Macro/positive F1.
- Route continuity (strict and relaxed).
- LCS sequence F1 and exact match.
- Number of impossible jumps fixed.
- Inference time.

Report improvements **and** regressions: wrong candidate set (HMM cannot recover), over-smoothing at a junction, bad transition parameters on long gaps, or a route whose reference sequence is themselves discontinuous.

**Tuning:** λ, σ, β, $v_\text{max}$, gap threshold and U-turn penalty are tuned on training/student-test data only. Store the grid/search in `results/hmm/decoder_tuning.csv`.

---

## 14. API design

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Model loaded? artifact version? trip count? |
| `/version` | GET | Git commit, model hash, package versions |
| `/trips` | GET | The ten evaluation trips (ID + metadata, no ground truth) |
| `/predict` | POST | Run the full pipeline for a trip; return route + metrics + timings |

**Request/response contract**

- `POST /predict` body: `{"trip_id": "00000000"}`.
- Response: `trip_id`, `model_version`, `metrics`, `predicted_route` (list of `[lon, lat]`), `observed_points` (optional), `n_observations`, `timings_ms`, `decoder` (`classifier_only` or `hmm`).
- **Never** include the reference route, per-observation truth labels, or the `.route` content.

**Safety properties**

- `trip_id` is validated (`^[0-9]{8}$`) and resolved inside the configured data root; no path traversal.
- The prediction path opens only `.track`, `.nodes`, `.arcs`. Scoring opens `.route` **after** prediction.
- Artifacts are loaded once at startup; a missing artifact yields a clear 503, not a silent fallback.
- CORS is configured via env; an optional `X-API-Key` protects the endpoint.
- Every prediction logs one structured line: request ID, trip, point count, candidate count, decoder, top-1, latency, model hash. This is the evidence that inference is live.
- No hardcoded outputs, no per-trip branches.

**Why FastAPI?** Typed request/response models, automatic validation and OpenAPI docs, async serving, easy CORS, and a small enough surface for a hackathon. Uvicorn runs it on Laptop 1.

**Latency budget** (fill with measured values): load/candidates/features/model/Viterbi/scoring per stage; aim for interactive use (< 30 s for the largest trip). The UI shows a spinner and the API returns a clear timeout/error if the budget is exceeded.

---

## 15. UI design

- Streamlit on Laptop 2, talking to the API via `PATHSNAP_API_URL`.
- Sidebar: API URL, health badge, model version.
- Main: select one of the ten trips → **Run prediction**.
- Results: metric cards (Top-1, macro F1, continuity, sequence F1), stage timings, number of observations.
- Map: GPS observations + predicted route, with layer control and a legend; zoomed to the trip’s extent.
- Raw JSON in an expander for transparency.
- Error states: API unreachable, model not ready, trip failed, timeout — each with an actionable message.

**What the UI must not do:** load `.route`, compute metrics client-side, import the model, or display the reference route. The evaluator sees the predicted route and the server-computed score.

---

## 16. Two-laptop deployment

### 16.1 Pre-flight

1. Both laptops on the same Wi-Fi/hotspot; client isolation off.
2. Laptop 1 has the repo, `data/evaluator_test/`, and `artifacts/`.
3. `pip install -r requirements.txt` on both (the UI only needs `streamlit`, `requests`, `folium`, but the full install is fine).
4. Find Laptop 1's IP: `hostname -I` / `ip addr` (Linux) or `ipconfig` (Windows).
5. Open port 8000 in the firewall.
6. `curl http://<LAPTOP1_IP>:8000/health` from Laptop 2.

### 16.2 Start

```bash
# Laptop 1
PATHSNAP_DATA_ROOT=./data uvicorn backend.app:app --host 0.0.0.0 --port 8000 --log-level info

# Laptop 2
PATHSNAP_API_URL=http://<LAPTOP1_IP>:8000 streamlit run ui/app.py --server.address 0.0.0.0
```

### 16.3 Demo evidence

- Terminal on Laptop 1 with the request log visible (timestamp, trip, latency, model hash).
- UI screenshot showing the selected trip, metrics, map and timings.
- A second screenshot of the API terminal after the evaluator’s request.
- Save both under `results/screenshots/` and include them in the report §7.

### 16.4 Contingencies

| Failure | Action |
|---|---|
| UI cannot reach API | Check IP, port, firewall, hotspot isolation; fall back to Laptop 1 running both processes and note the limitation |
| Model artifact missing | Stop; export the artifact; never fake a response |
| A trip fails | Show the error, capture the log, retry once; report the trip and the cause |
| API slow on a large trip | Explain the stage timings; use a warmed-up cache |
| Network dies mid-demo | Keep the API terminal visible; restart the UI client; do not restart the model |

---

## 17. Folium visualisation

Required layers:

1. Original GPS observations (markers or a thin line).
2. Road-network geometry in the trip’s extent (light lines).
3. Reference route, where available (for training/student-test report maps only).
4. Predicted route (distinct colour).
5. Layer control + legend; appropriate zoom and extent.

Deliverables:

- `results/maps/<trip_id>.html` committed.
- At least one screenshot in the report PDF.
- The map-generation code (`pathsnap/visualize.py`, `visualisation/`) committed.

Do not fabricate maps: generate them from the actual pipeline outputs. Do not include the reference route in the **evaluation UI**; report maps may show it because they are generated offline from permitted data.

---

## 18. Numbers ledger (single source of truth)

Fill every row from the cited file; update in the same PR as the experiment.

| Claim | Value | Source | Owner | Status |
|---|---|---|---|---|
| Trips in train / student-test | 80 / 10 | `results/eda/inventory.csv` | A | verified |
| Train / student-test observations | 208,417 / 19,160 | `results/eda/summary.json` | A | verified |
| Candidate miss rate @ 60 m | 0.092% train / 0.032% student-test (60 m + fallbacks, top-K 15; 8.7 candidates/obs) | `results/eda/candidate_sweep.csv` | A | verified |
| Label coverage (train) | mean 96.8% per trip, median 99.87%, pooled 87.8% | `results/labels/summary.json` | A | verified |
| Labelled observations (train) | 183,052 / 208,417 | `results/labels/summary.json` | A | verified |
| Candidate miss rate vs top-K (train, any radius ≥ 30 m) | K=5 ≈1.2%, K=10 ≈0.24%, K=15 ≈0.09%, K=30 ≈0.002% | `results/eda/candidate_sweep.csv` | A | verified |
| Reference trips flagged unreliable | 2 (train `00000033`, `00000054`); 0 in student-test | `results/labels/coverage.csv` | A | verified |
| Dataset candidate rows (train) | TBD | `results/models/dataset_summary.json` | B | TBD |
| Best CV Top-1 (model) | TBD | `results/models/model_comparison.csv` | B | TBD |
| Best CV macro F1 | TBD | `results/models/model_comparison.csv` | B | TBD |
| Student-test Top-1 (selected) | TBD | `results/models/model_comparison.csv` | B | TBD |
| Student-test log loss | TBD | `results/models/model_comparison.csv` | B | TBD |
| Best model + config | TBD | `results/models/selected_model.json` | B | TBD |
| Optuna trials / best CV score | TBD | `results/optuna/trials.csv` | B | TBD |
| Tuned vs untuned delta | TBD | `results/models/tuned_vs_untuned.csv` | B | TBD |
| Decoder best (λ, σ, β, gap) | TBD | `results/hmm/decoder_tuning.csv` | C | TBD |
| HMM Top-1 vs classifier-only | TBD | `results/hmm/classifier_vs_hmm.csv` | C | TBD |
| HMM continuity vs classifier-only | TBD | `results/hmm/classifier_vs_hmm.csv` | C | TBD |
| HMM sequence LCS F1 | TBD | `results/hmm/classifier_vs_hmm.csv` | C | TBD |
| API p50 / p95 latency | TBD | `results/integration/latencies.csv` | D | TBD |
| Student-test trips passing 10/10 | TBD | `results/integration/run_log.csv` | D | TBD |
| Model artifact SHA-256 | TBD | `artifacts/model_card.json` | D | TBD |

---

## 19. Likely viva questions (50)

### Data and EDA — mostly A

1. **What does each file contain?** Track = noisy GPS + timestamps; nodes = road-network coordinates; arcs = directed node pairs (arc ID = row index); route = chronological reference arc IDs.
2. **Why is a `.route` row not a GPS point?** A route is a sequence of traversed arcs; many GPS points project onto one arc.
3. **How many trips in each partition?** 80 train, 10 student-test, 10 evaluator.
4. **Why do different trips have different networks?** Each trip ships its own local subgraph extracted around the trajectory; there is no shared global graph.
5. **What are the main data-quality issues?** GPS noise, irregular sampling and gaps, long contracted arcs, partial/offset reference routes, zero-length arcs and disconnected components.
6. **How did you detect a bad reference route?** Projection distance along the trip, route coverage vs track extent, and progress plateau diagnostics; flagged trips are documented.
7. **How did you measure candidate miss rate?** Labeled observations whose aligned reference arc is absent from the candidate set; report vs radius and per trip.
8. **What is the sampling rate?** Nominally 1 Hz, but ~3% of steps differ and some trips have very large gaps.
9. **Did you use the evaluator data anywhere?** No. Not for training, tuning, thresholds, feature choice or model selection.
10. **How do you prevent trip leakage?** Grouped splits by trip; no observation from a trip appears in two folds or partitions.

### Geo and preprocessing — mostly A

11. **Why not Euclidean distance in degrees?** Degrees are angular; a degree of longitude shrinks with latitude. Use a local metric projection or geodesic distance.
12. **What projection did you use?** Local equirectangular around each trip’s mean latitude; error is negligible at trip scale.
13. **How do you compute point-to-segment distance?** Project the point onto the segment, clip the projection ratio to [0,1], measure the Euclidean distance in metres.
14. **How do you handle zero-length arcs?** Treat the distance as the endpoint distance; guard the denominator.
15. **Why smooth headings and speeds?** 1 Hz GPS jitter makes raw bearings unreliable; a centred window reduces noise.
16. **Does smoothing leak future information?** It uses neighbouring observations within the same trip only; it is applied identically at training and inference and does not use labels.
17. **What coordinate order do the files use?** Longitude, latitude. Reversing it would put the data in the ocean; the loader validates bboxes.

### Candidates, labels, features — mostly A/B

18. **Why generate candidates instead of classifying all arcs?** The full network is huge; a shortlist makes training and inference tractable while keeping recall measurable.
19. **How does the KD-tree work here?** We index dense samples along each arc, query a radius around each GPS point, deduplicate arcs, then compute exact distances for the shortlist.
20. **Why not index only arc midpoints?** Long arcs would be missed; a midpoint far from the point can still have a nearby endpoint. Dense sampling avoids that.
21. **What happens if there are no candidates?** Expand the radius and retry; record the fallback; if still empty, mark the observation as unmatched.
22. **How are labels generated?** Project each GPS point onto the reference-route polyline, enforce monotonic progress, map to the arc, and flag low-confidence points.
23. **Why monotonic progress?** A vehicle can stop but cannot teleport backwards along the route; enforcing monotonicity removes projection ambiguity.
24. **What if the route revisits an arc?** Nearest-segment alignment can be ambiguous; a DP/windowed alignment resolves it, and ambiguous points are flagged.
25. **Why exclude low-confidence labels from training?** A bad label teaches the wrong arc. We report coverage rather than silently using noise.
26. **Which features matter most?** Fill from the actual importance plot; distance, heading agreement and graph continuity are the expected leaders.
27. **Why no node/arc ID features?** They are trip-local and would not generalise to unseen networks.
28. **Why log-transform features?** Distances and lengths are heavy-tailed; logs stabilise linear models and neural networks.
29. **Did you ablate the features?** Yes: local → + relative → + temporal/graph; report CV Top-1 and F1 for each group.

### Models and evaluation — mostly B

30. **Why at least five models?** The brief requires it, and different families expose different bias/variance behaviour; selection must be evidence-based.
31. **What is your primary selection metric and why?** CV observation Top-1 on fixed candidate sets; it measures the actual decision. Macro F1 and log loss are guardrails.
32. **Why not use accuracy?** Candidate pairs are imbalanced; always predicting “negative” scores high accuracy while being useless.
33. **What is macro F1?** The unweighted mean of positive and negative F1; it exposes poor performance on the minority class.
34. **Why GroupKFold?** Observations from the same trip are correlated; random folds leak trip-specific patterns.
35. **How do you handle class imbalance?** Class weights and negative subsampling during training; evaluation always uses full candidate sets.
36. **How do you know a model overfits?** Train vs validation gap, fold variance, and (only where real) epoch-wise learning curves.
37. **Which model won and why?** Fill from the comparison; justify with generalisation, probability quality, cost and decoder compatibility.
38. **Is the linear model useful if it loses?** Yes: it is an interpretable baseline and shows how much nonlinearity is needed.
39. **Why calibrate probabilities?** HMM emissions and log loss require probabilities that mean something, not arbitrary scores.
40. **What is the inference cost?** Report per-stage and per-trip timings from the actual runs.

### Bayesian optimisation — B

41. **Why Bayesian optimisation?** Each trial is expensive (grouped CV); TPE learns from previous trials and proposes better configurations.
42. **What is the objective?** Mean CV Top-1 over 5 grouped folds, with F1/log loss as tie-breakers.
43. **What search space did you use?** List the actual ranges from `results/optuna/`.
44. **How many trials?** The actual count and wall-clock cost.
45. **Did tuning help?** Report the measured delta on CV and student-test; a negative result is acceptable.
46. **Why not tune on the student-test set?** It is a development comparison set; tuning on it would make the reported result optimistic and violate the protocol.
47. **How do you avoid overfitting the search?** Grouped folds, a bounded search space, a fixed trial budget, and a final check on held-out student-test trips.

### HMM and system — C/D

48. **What are the hidden states, emissions and transitions?** States = candidate arcs; emissions = calibrated classifier probabilities (optionally distance-blended); transitions = network-distance consistency with GPS displacement.
49. **Write the Viterbi recurrence.** $\delta_t(j)=\log\tilde e_t(j)+\max_i[\delta_{t-1}(i)+\log a_{ij}]$, with backpointers; initialise with the start prior and backtrack from the best final state.
50. **Did the HMM help?** Compare Top-1, continuity and LCS F1 on student-test; show a case where it fixed a jump and a case where it hurt.
51. **How are gaps handled?** Skip the state layer or reset the chain when the time gap is large; document the threshold and tune it on permitted data.
52. **What if a candidate is missing at one observation?** Carry the previous state forward and mark the observation unmatched; do not invent an arc.
53. **Can the HMM fix a missing truth arc?** No. Missing candidates are a hard ceiling; that is why the miss rate is reported.
54. **How does the UI get the score if ground truth is server-side?** The API computes it after prediction and returns only the metrics; the UI never sees the route.
55. **How do you prove the API is live?** The terminal log line per request and the model hash; evaluators can watch the request arrive.
56. **What is the two-laptop architecture?** Laptop 1 = model/API/HMM/scoring/ground truth; Laptop 2 = thin Streamlit client over HTTP.
57. **What would you do with more time?** Better route-quality detection, bidirectional/beam decoding, richer topology features, calibrated per-trip thresholds, and a larger candidate set with learned pruning.

*(The numbering above intentionally exceeds 50; the full bank is in §19.1 below for cross-examination.)*

### 19.1 Rapid-fire one-liners

| Question | Answer |
|---|---|
| What is map matching? | Assigning noisy GPS observations to road-network segments and reconstructing a connected route. |
| Why is it hard? | Noise, low sampling, gaps, parallel roads, junctions, one-way rules, long chords. |
| What is an emission? | The likelihood of an observation given a hidden road segment. |
| What is a transition? | The likelihood of moving between two road segments, constrained by the network and time. |
| Why log space? | Many small probabilities underflow when multiplied. |
| Complexity of Viterbi? | $O(TK^2)$ transitions plus shortest-path computation. |
| What is candidate miss rate? | The fraction of truths absent from the candidate set. |
| What is route continuity? | The fraction of consecutive predictions that are connected in the road graph. |
| What is LCS F1? | Order-aware precision/recall of the predicted arc sequence against the reference. |
| Why keep ground truth server-side? | Integrity: the model and UI must not see the answer. |
| What is the model input? | A vector of relative geometric/topological features for an observation–candidate pair. |
| What is the model output? | A calibrated probability that the candidate is the correct arc. |
| What does the HMM add? | Sequence consistency and network feasibility. |
| What is the biggest failure mode? | A truth arc missing from the candidate set, or a misaligned reference route. |

---

## 20. Failure analysis to prepare (honest, concrete)

Prepare at least one example for each:

1. **Candidate miss** — the reference arc is outside the radius; the model cannot recover. Show the map and the miss-rate row.
2. **Parallel-road confusion** — two nearby roads with similar distance and heading; distance alone fails. Show how graph continuity or HMM fixes or fails.
3. **Large GPS gap** — the vehicle moves far between observations; the transition constraint must relax. Show before/after.
4. **Disconnected prediction** — classifier-only produces a jump; HMM restores continuity. Show the continuity metric.
5. **Over-smoothing** — HMM keeps the vehicle on the wrong but connected road, or suppresses a legitimate turn. Show the sequence-F1 drop.
6. **Reference-route limitation** — a partial/offset route makes the pointwise score look worse than the prediction. Show the flagged trip and explain.

The evaluator will value a clear “why it fails” over a claim that everything works.

---

## 21. Per-member must-know (minimum)

| Member | Must be able to explain without notes |
|---|---|
| A | File semantics; inventory numbers; projection; candidate algorithm; label alignment; flagged trips; Folium map code |
| B | Feature list and rationale; 5+ models; CV protocol; metric formulas; selection decision; Optuna space/objective/results |
| C | HMM state/emission/transition; Viterbi recurrence; gap handling; transition graph implementation; classifier-vs-HMM results |
| D | API endpoints and schemas; ground-truth isolation; two-laptop networking; logging evidence; UI flow; demo contingencies |

Cross-training: each member also presents one section they did **not** own in the mock viva.

---

## 22. Final checklist (T-60 minutes)

### Code and repo

- [ ] `git status` clean; all source committed; no notebooks with the only copy of a result.
- [ ] `make test` passes on a clean clone.
- [ ] `make smoke` passes.
- [ ] `requirements.txt` installs without the raw dataset.
- [ ] `artifacts/model.joblib`, feature config and HMM config committed (subject to size limits).
- [ ] No secrets/API keys: `grep -RIn "api_key\|password\|token" --exclude-dir=.git .`
- [ ] Repo is public; check in an incognito window.
- [ ] README setup/run instructions match the actual commands.
- [ ] `data/` and course PDFs are not committed (or are intentionally handled).

### Model and decoder

- [ ] Selected model loads from `artifacts/` and returns the same metrics as the training report.
- [ ] Tuned vs untuned table complete.
- [ ] HMM config matches the reported tuning.
- [ ] All 10 student-test trips run through the API without manual intervention.
- [ ] Ground-truth isolation test passes.
- [ ] No hardcoded trip outputs; API log shows live inference.

### API/UI/demo

- [ ] Laptop 1 and Laptop 2 on the same network; `curl /health` works.
- [ ] UI lists the ten evaluator trips (or the fallback, clearly labelled).
- [ ] Prediction completes within the budget; spinner/error states work.
- [ ] Terminal logs visible and readable; model hash printed.
- [ ] Screenshots of UI and API logs saved.
- [ ] Demo script rehearsed end-to-end at least once.

### Viva and report

- [ ] `viva.md` has no `TBD` left for reported numbers.
- [ ] Every number in the report appears in the ledger with a source.
- [ ] Each member has presented their own section and one foreign section.
- [ ] Failure cases ready with maps.
- [ ] Report PDF complete, page numbers, figures referenced, AI disclosure included.

---

## 23. Demo script (5–7 minutes)

1. **Context (30 s):** “GPS is noisy; the task is to recover the road sequence. We serve it from two laptops; the UI never sees ground truth.”
2. **Show the UI (30 s):** health badge, trip list, select one of the ten trips.
3. **Run inference (30 s):** click Run; point to the spinner and the API terminal log line (timestamp, trip, latency).
4. **Show results (60 s):** metric cards, predicted route on the map, observed GPS, layer control.
5. **Explain the pipeline (90 s):** candidates → features → classifier → HMM → Viterbi; point to the code paths.
6. **Show evidence (60 s):** model comparison table, Optuna history, classifier-vs-HMM table, one failure map.
7. **Honesty (30 s):** candidate miss rate, flagged reference routes, what we would improve.
8. **Invite questions (rest):** use the Q&A bank; if stuck, say what you would measure.

**Never** start the demo with a slide deck if the evaluator asked for a live system. Show the system first, then the evidence.

---

## 24. AI-assistance disclosure (draft)

> During this project, AI tools were used to assist with repository scaffolding, code review suggestions, debugging, and language editing. All model training, hyperparameter searches, decoder experiments, measurements and visualisations were executed by the team. Every team member can explain the submitted code, experimental results and design decisions. No reported metric was generated without a corresponding experiment artifact in the repository.

Adjust to the actual tools and scope before submission. Do not claim that AI-generated analysis was independently performed experimentation.
