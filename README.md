# PathSnap — Map-Matching & Route Reconstruction from Noisy GPS

**PathSnap** reconstructs the road route a vehicle actually drove from a noisy,
1 Hz GPS trace. Each GPS point is matched to a road segment of the trip's own
road network with a learned candidate classifier, and a network-aware
**HMM/Viterbi** decoder turns those per-point decisions into one connected,
drivable route. The system is deployed across **two laptops**: a FastAPI service
(model, decoder, road networks, server-side scoring) and a Streamlit UI.

*UE24CS352A Machine Learning — Group 8.*

## Results at a glance

On the **10 student-test trips**, which were never used for training or for
choosing any setting (mean per trip, `results/hmm/classifier_vs_hmm.csv`):

| | Classifier only | **Classifier + HMM (final)** |
|---|---|---|
| **Point accuracy (Top-1)** | 0.900 | **0.930** |
| Route continuity (strict) | 0.755 | **0.906** |
| Sequence LCS F1 | 0.772 | **0.933** |
| Arc-set F1 | 0.889 | **0.938** |
| Worse than classifier-only on | — | **0 of 10 trips** |

- Best classifier: **HistGradientBoosting**, chosen from **8 model families** —
  5-fold trip-grouped CV Top-1 **0.887**, macro F1 **0.928**; student-test
  Top-1 **0.882**, macro F1 **0.925** (classifier-only, pooled over points).
- Snapping every point to the nearest road scores only **0.575**.
- The live API answers in **563 ms** median (p95 1.67 s) per trip in HMM mode.

---

## Contents

1. [The problem and the data](#1-the-problem-and-the-data)
2. [How it works](#2-how-it-works)
3. [Results in detail](#3-results-in-detail)
4. [Quick start](#4-quick-start)
5. [Reproduce every result](#5-reproduce-every-result)
6. [Final demo on two laptops](#6-final-demo-on-two-laptops)
7. [API reference](#7-api-reference)
8. [Repository layout](#8-repository-layout)
9. [Documentation](#9-documentation)

---

## 1. The problem and the data

GPS fixes drift several metres, jump across buildings and drop out, so the raw
trace rarely lies on a road. The task is to output, for every GPS point, the
road segment (arc) the vehicle was on, and a route that is actually connected.

Each trip is a folder of four headerless text files:

| File | Contents | Note |
|---|---|---|
| `.track` | longitude, latitude, timestamp (s) | noisy GPS, nominally 1 Hz |
| `.nodes` | longitude, latitude | node ID = row index |
| `.arcs` | from-node, to-node | directed road segment, arc ID = row index |
| `.route` | arc IDs in driving order | reference route — used only for scoring |

| Partition | Trips | GPS points | Used for |
|---|---|---|---|
| train | 80 | 208,417 | fitting, cross-validation, all tuning |
| student-test | 10 | 19,160 | held-out evaluation only |
| evaluator | 10 | — | final demo only; never read during development |

What the data analysis found (`results/eda/`, `docs/DATA_NOTES.md`): no
missing or invalid GPS rows; 96.7% of steps exactly 1 s apart but gaps up to
42 hours; trips spread across several continents, each with its own network
(299–155,003 nodes); arcs are straight chords up to 9.3 km long; 11 reference
routes have breaks and 2 train trips have unreliable references.

## 2. How it works

```
GPS trace + trip road network
   │
   ▼
1. Project to local metres, smooth (5-point window) → heading and speed
   ▼
2. Candidates: KD-tree over arc samples every 5 m → ≤ 15 arcs within 60 m per point
   ▼
3. Features: 18 trip-relative numbers per (point, candidate)
   ▼
4. Classifier (HistGradientBoosting) → probability per candidate
   ▼
5. HMM + Viterbi: emissions = classifier probabilities,
   transitions = network distance vs GPS displacement → one connected route
   ▼
6. Server-side scoring against the reference route (never seen by steps 1–5)
```

**Key design choices**

- **No coordinates or IDs as features.** Every trip is a different city, so all
  18 features are relative: distance to the arc, heading difference, position
  along the arc, distance ratio to the closest candidate, junction degree,
  network distance from the previous point, speed and time gap.
- **Labels from the reference route.** Each GPS point is labelled with the
  nearest reference-route arc if it lies within 30 m; farther points carry no
  label (`docs/DECISIONS.md` D-009).
- **Trip-grouped cross-validation.** Five folds split by trip, so a model is
  always tested on drives it has never seen.
- **HMM transitions use the road graph.** Moving between candidates must be
  reachable at a plausible speed; the cost compares network distance with GPS
  displacement. Same-arc backward GPS jitter up to 40 m is tolerated and
  U-turns are penalised — the change that made the HMM beat the classifier.
- **Ground truth stays on the server.** The prediction path never opens a
  `.route` file and the UI never receives the reference route or labels.

## 3. Results in detail

### 3.1 Candidate generation and labels

| Metric | Value | Source |
|---|---|---|
| Points whose true arc is missing from the candidates (train / student-test) | 0.092% / 0.032% | `results/eda/candidate_sweep.csv` |
| Mean candidates per point | 8.7 | `results/eda/candidate_sweep.csv` |
| Labelled train points (within 30 m of the reference) | 183,052 of 208,417 | `results/labels/summary.json` |
| Training rows (point × candidate) | 1,732,459 | `results/models/dataset_summary.json` |

### 3.2 Classifier comparison (8 families, default settings)

| Model | CV Top-1 | CV macro F1 | CV log loss | Student-test Top-1 | Student-test macro F1 |
|---|---|---|---|---|---|
| **HistGradientBoosting** (selected) | **0.887** | **0.928** | **0.084** | **0.882** | **0.925** |
| Random forest | 0.886 | 0.923 | 0.098 | 0.879 | 0.916 |
| Extra Trees | 0.885 | 0.917 | 0.115 | 0.867 | 0.906 |
| Decision tree | 0.879 | 0.898 | 0.142 | 0.866 | 0.884 |
| MLP (64-32) | 0.878 | 0.922 | 0.096 | 0.876 | 0.918 |
| Linear SVM (calibrated) | 0.871 | 0.898 | 0.123 | 0.845 | 0.891 |
| Logistic regression | 0.871 | 0.857 | 0.179 | 0.844 | 0.851 |
| Gaussian Naive Bayes | 0.845 | 0.137 | 4.746 | 0.833 | 0.121 |
| Nearest-road baseline | 0.575 | — | — | 0.564 | — |

CV = 5-fold GroupKFold over the 80 train trips; student-test = refit on all
train trips (`results/models/cv_vs_student_test.csv`, `report/03_models.md`).
HistGradientBoosting wins on Top-1 and on probability quality, which matters
because the HMM consumes the probabilities.

**Feature ablation** (`results/models/feature_ablation.csv`): distance alone
0.567 → local geometry 0.879 → all 18 features 0.887. Removing just the two
heading features drops Top-1 to 0.646 — direction of travel is the key signal.

### 3.3 Bayesian hyperparameter optimisation

40 Optuna TPE trials over learning rate, trees, leaves, minimum leaf size and
L2, each scored with the same 5-fold trip-grouped CV (35 minutes). Best CV
Top-1 0.8872 vs 0.8868 for the defaults — inside fold noise, with slightly
worse F1 and log loss — so the **default configuration was kept**
(`results/optuna/`, `results/models/tuned_vs_untuned.csv`, `report/04_optuna.md`).

### 3.4 HMM decoder

| Student-test (held out) | Classifier only | HMM, first tuning | **HMM, final tuning** |
|---|---|---|---|
| Top-1 | 0.900 | 0.852 | **0.930** |
| Strict continuity | 0.755 | 0.873 | **0.906** |
| LCS F1 | 0.772 | 0.720 | **0.933** |

The first tuning grid could not stop the HMM losing accuracy: GPS jitter at
stops made "stay on this arc" infeasible, so the decoder flipped to the
opposite-direction arc of the same road. The final tuning adds a same-arc
backward tolerance, selects the settings on **24 train trips** using
out-of-fold classifier probabilities, and reports on the untouched
student-test trips. Final settings (`artifacts/hmm_config.json`): tolerance
40 m, β 5, U-turn penalty 40, λ 0, gap reset 60 s
(`results/hmm/decoder_tuning_v2_refine2_best.json`, `report/05_hmm.md`).

### 3.5 End-to-end through the API

All 10 student-test trips run through the real API code path in both modes
(`results/integration/summary.json`, `results/integration/stage_timings.csv`):

| Mode | Trips passed | Top-1 | Continuity | Latency p50 / p95 | Worst trip |
|---|---|---|---|---|---|
| Classifier only | 10/10 | 0.900 | 0.755 | 208 / 531 ms | 0.62 s |
| HMM (final) | 10/10 | 0.930 | 0.906 | 563 / 1669 ms | 1.85 s |

Interactive classifier-vs-HMM maps: `results/maps/`; report figures:
`results/figures/`.

## 4. Quick start

Requires Python 3.11+ (tested on 3.14) and about 1 GB of disk for the
environment and caches.

```bash
git clone https://github.com/Siddhesh249/ML-HACKATHON-Group-8.git
cd ML-HACKATHON-Group-8
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make test                          # ~40 tests, no dataset needed
```

**Place the dataset** (not committed — course material):

```
data/
├── train/<trip_id>/<trip_id>.{track,nodes,arcs,route}           # 80 trips
├── student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}    # 10 trips
└── evaluator_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}  # 10 trips, demo laptop only
```

Copying or symlinking both work, e.g.
`ln -s "/path/to/Student folder/train" data/train`.

**Run it on one machine** (two terminals):

```bash
make api                                            # terminal 1 → http://localhost:8000
PATHSNAP_API_URL=http://localhost:8000 make ui      # terminal 2 → http://localhost:8501
```

The trained model (`artifacts/model.joblib`) and decoder settings
(`artifacts/hmm_config.json`) are committed, so no training is needed to run
the system.

## 5. Reproduce every result

Each step writes to `results/`; run them in order from the repository root.

```bash
python3 scripts/01_eda.py --data-root ./data                 # data inventory, quality checks, figures
python3 scripts/01b_candidate_sweep.py --data-root ./data    # candidate radius / top-K sweep
python3 scripts/01c_label_coverage.py --data-root ./data     # label coverage + visual checks
python3 scripts/02_build_dataset.py --data-root ./data       # training rows → cache/dataset.npz
python3 scripts/02_build_dataset.py --data-root ./data --partitions student_test --out cache/student_test.npz
python3 scripts/03_train_models.py --folds 5 --profile full  # 8-model comparison (~15 min)
python3 scripts/03b_feature_ablation.py                      # feature ablation + importance
python3 scripts/04_tune.py --trials 40 --folds 5             # Optuna (~35 min)
python3 scripts/04b_student_test_eval.py --data-root ./data  # student-test + tuned vs untuned
python3 scripts/04c_model_figures.py                         # comparison bars, learning curves
python3 scripts/06_export_artifacts.py                       # artifacts/model.joblib
python3 scripts/05d_tune_hmm_v2.py --data-root ./data        # HMM tuning on train trips (~5 min)
python3 scripts/05_eval_hmm.py --data-root ./data            # classifier vs HMM, student-test
python3 scripts/06_hmm_failure_analysis.py --data-root ./data
python3 scripts/run_eval.py --data-root ./data --partition student_test --modes hmm classifier
python3 scripts/summarise_integration.py                     # integration summary + timings
python3 visualisation/make_maps.py --data-root ./data        # Folium maps + screenshots
python3 scripts/audit_claims.py                              # every reported number vs its file
```

Seeds are fixed (`PATHSNAP_SEED=42`); the evaluator partition is never read by
any of these scripts.

## 6. Final demo on two laptops

**Laptop 1** runs the API: model, HMM, road networks, the 10 evaluator trips and
server-side scoring. **Laptop 2** runs only the UI. The evaluator picks a trip in
the UI, Laptop 1 computes and scores the route live, and its terminal log shows
each request arriving.

### Step 1 — prepare both laptops (the day before)

On **both** laptops:

```bash
git clone https://github.com/Siddhesh249/ML-HACKATHON-Group-8.git
cd ML-HACKATHON-Group-8
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On **Laptop 1** only, place the 10 evaluator trips and check the artifacts:

```bash
ls data/evaluator_test | wc -l                          # expect 10
ls artifacts/model.joblib artifacts/hmm_config.json
make test
```

Laptop 2 needs **no data and no `.route` files**.

### Step 2 — connect the network

1. Put both laptops on the **same Wi-Fi or phone hotspot** (turn off "client
   isolation" / "AP isolation" if the hotspot has it).
2. Find Laptop 1's IP address:
   - macOS: `ipconfig getifaddr en0`
   - Linux: `hostname -I`
   - Windows: `ipconfig` → "IPv4 Address"
3. Allow incoming connections on port **8000** on Laptop 1 (macOS: allow Python
   when the firewall asks; Linux: `sudo ufw allow 8000/tcp`; Windows: allow
   Python on private networks).

### Step 3 — start the API on Laptop 1

```bash
cd ML-HACKATHON-Group-8 && source .venv/bin/activate
make api
# same as: PATHSNAP_DATA_ROOT=./data python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --log-level info
```

Keep this terminal **open and visible** — the evaluator must see the requests.

### Step 4 — start the UI on Laptop 2

```bash
cd ML-HACKATHON-Group-8 && source .venv/bin/activate
curl http://<LAPTOP1_IP>:8000/health          # expect "ready": true
PATHSNAP_API_URL=http://<LAPTOP1_IP>:8000 make ui
# Windows PowerShell: $env:PATHSNAP_API_URL="http://<LAPTOP1_IP>:8000"; make ui
#                     (or: python -m streamlit run ui/app.py --server.port 8501)
```

Open `http://localhost:8501` on Laptop 2.

### Step 5 — run the demo

1. The UI lists the **10 evaluator trips** and shows a green *Ready* status.
2. The evaluator selects any trip and presses **Run prediction** (HMM mode).
3. Laptop 1's terminal prints the request and the server-side score:
   ```
   HTTP POST /predict status=200 ms=... rid=...
   PREDICT trip=... points=... candidates=... mode=hmm top1=... continuity=... latency_ms=...
   ```
4. The UI shows accuracy, continuity and sequence F1, the predicted route on a
   map and the stage timings.
5. Run a second trip to show the results are computed live, not hard-coded.
   Switching to classifier-only mode on the same trip shows what the HMM adds.

### If something goes wrong

| Symptom | Fix |
|---|---|
| UI says *API unreachable* | wrong IP, firewall, or different network — re-run `curl http://<LAPTOP1_IP>:8000/health` from Laptop 2 |
| `/health` shows `ready: false` | `artifacts/model.joblib` missing — `git pull` or `python3 scripts/06_export_artifacts.py` |
| UI lists student-test trips, not evaluator trips | `data/evaluator_test/` is missing on Laptop 1 |
| `404 Trip not found` | the trip folder name is not the 8-digit ID |
| CORS error | start the API with `PATHSNAP_CORS_ORIGINS=*` |
| Network fails completely | run the UI on Laptop 1 with `PATHSNAP_API_URL=http://localhost:8000` and explain the fallback |

Full runbook with checklists and contingencies: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

### Configuration

| Variable | Default | Used by |
|---|---|---|
| `PATHSNAP_DATA_ROOT` | `./data` | API — folder containing the partitions |
| `PATHSNAP_EVALUATION_PARTITION` | `evaluator_test` | API — trips offered in the UI |
| `PATHSNAP_FALLBACK_PARTITION` | `student_test` | API — used when evaluator trips are absent |
| `PATHSNAP_ARTIFACT_DIR` | `./artifacts` | API — model and decoder settings |
| `PATHSNAP_API_URL` | `http://localhost:8000` | UI — where the API runs |
| `PATHSNAP_CORS_ORIGINS` | `*` | API — allowed UI origins |
| `PATHSNAP_API_KEY` | empty | API — optional `X-API-Key`; never commit a value |

`.env.example` lists them all.

## 7. API reference

| Endpoint | Method | Returns |
|---|---|---|
| `/health` | GET | ready flag, model name and version, trip count, partition |
| `/version` | GET | API version, git commit, Python and model version |
| `/trips` | GET | the evaluation trips (IDs, sizes, bounding boxes — no ground truth) |
| `/predict` | POST | `{"trip_id": "00000000", "mode": "hmm"}` → metrics, predicted route, observed points, stage timings |

`mode` is `hmm` (default) or `classifier`. The response never contains the
reference route or per-point labels.

## 8. Repository layout

```
pathsnap/        core library: loading, geometry, candidates, labels, features,
                 models, metrics, HMM/Viterbi, scoring, Folium maps
backend/         FastAPI service (Laptop 1)
ui/              Streamlit client (Laptop 2)
scripts/         one script per pipeline stage (see §5)
visualisation/   map generation
artifacts/       committed model, feature contract, HMM settings, model card
results/         every reported number and figure (CSV / JSON / PNG / HTML)
report/          technical report sections
tests/           unit, contract and ground-truth-leakage tests
docs/            data notes, experiment protocol, HMM design, deployment runbook
data/            dataset (not committed)
```

## 9. Documentation

| Document | Contents |
|---|---|
| [`report/`](report/) | technical report sections (models, optimisation, HMM, deployment) |
| [`viva.md`](viva.md) | explanations, numbers ledger, Q&A bank, demo script |
| [`docs/DATA_NOTES.md`](docs/DATA_NOTES.md) | dataset facts and quirks |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | experimental protocol and metric definitions |
| [`docs/HMM_DESIGN.md`](docs/HMM_DESIGN.md) | HMM states, emissions, transitions, Viterbi |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | design decisions and why |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | full two-laptop runbook |
| [`results/audit/README.md`](results/audit/README.md) | check that every reported number matches a results file |
| [`docs/TEAM.md`](docs/TEAM.md) | team, task board and working agreement |
| [`AGENTS.md`](AGENTS.md) | contributor and agent protocol |
