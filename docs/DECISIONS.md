# Decisions (Architecture Decision Records)

> **Append-only.** Each decision is immutable once accepted. To change a
> decision, add a new ADR that supersedes it and update the index. Do not rewrite
> history; the viva may ask why a choice was made and when it changed.

## Index

| ID | Title | Status | Date | Owner |
|---|---|---|---|---|
| D-001 | Two-stage candidate classification + HMM decoding | Accepted | 2026-11-18 | A/B/C |
| D-002 | Per-trip local metric projection | Accepted | 2026-11-18 | A |
| D-003 | Group-aware cross-validation by trip | Accepted | 2026-11-18 | B |
| D-004 | Ground truth isolated from prediction and UI | Accepted | 2026-11-18 | D |
| D-005 | Python/scikit-learn stack; no deep learning, no XGBoost/LightGBM | Accepted | 2026-11-18 | B |
| D-006 | Raw dataset not committed; artifacts and results are | Accepted | 2026-11-18 | D |
| D-007 | Append-only agent log + live status + ADRs | Accepted | 2026-11-18 | D |
| D-008 | Shared `pathsnap` library with thin API/UI | Accepted | 2026-11-18 | D |
| D-009 | Labels use the nearest route arc; monotonicity is a diagnostic | Accepted | 2026-11-18 | A/B |

---

## D-009 — Labels use the nearest route arc; monotonicity is a diagnostic

**Context:** the reference route is a sequence, but several trips are loops,
out-and-back routes, or ordered opposite to the GPS trace. Enforcing monotonic
arclength progress assigned labels to distant segments and produced false
“unreliable route” flags on 13 trips (coverage as low as 1.3%).

**Decision:** for each observation, label it with the arc of the **nearest route
segment**, subject to a maximum projection distance (30 m). Compute monotonic
progress and the backward fraction only as diagnostics. Sequence-level metrics
use order-aware LCS and order-agnostic arc-set F1 so direction/loop ambiguity is
visible instead of hidden.

**Consequences:** coverage rose from 81.7% to 96.8% mean, labelled observations
from 158,760 to 183,052, with only two training routes genuinely unreliable
(`00000033`, `00000054`). Per-point labels are robust to loops and reversed
routes. The remaining risk is a route that passes twice near the same point on
different arcs; the nearest-arc rule picks the closer geometry, and the
confidence threshold plus sequence metrics expose the ambiguity.

**Alternatives rejected:** monotonic assignment (false failures on loops),
manual per-trip fixes (not reproducible, hides data limitations), dropping all
loop/reversed trips (loses valid supervision).

**Evidence note:** the scaffold-run numbers in this ADR were exploratory and
their output files were removed to restore ownership. A must reproduce the
coverage, labelled counts and flagged trips in T12 before they are quoted.

---

## D-001 — Two-stage candidate classification + HMM decoding

**Context:** hundreds of thousands of arcs per trip; noisy 1 Hz observations;
predictions must form a connected route. End-to-end sequence models are
data-hungry and hard to calibrate; nearest-arc snapping produces teleports.

**Decision:** stage 1 generates a shortlist of nearby arcs per observation and a
classifier scores each candidate. Stage 2 runs an HMM over candidate arcs with
classifier probabilities as emissions and network-distance consistency as
transitions; Viterbi decodes the route.

**Consequences:**
- Training is a standard tabular classification problem with measurable
  precision/recall/log loss.
- Candidate miss rate is a hard, reportable ceiling on accuracy.
- The decoder can be tuned and compared separately from the classifier.
- The UI/API can expose both classifier-only and HMM modes for the report.

**Alternatives rejected:** point regression (no road ID output), nearest-arc
snapping (no sequence context), end-to-end deep sequence model (data size,
calibration, explainability, time budget).

---

## D-002 — Per-trip local metric projection

**Context:** longitude/latitude are angular; Euclidean degrees are not metres.
Trips span a few tens of kilometres and different cities, so a single global
projection is awkward.

**Decision:** project each trip to a local tangent plane around its mean
latitude: `x=(lon-lon0)*111320*cos(lat0)`, `y=(lat-lat0)*110540`.

**Consequences:** distances are in metres, geometry is vectorised and fast, error
is <0.5% at trip scale. A geodesic/haversine function remains available for
verification and tests.

**Alternatives rejected:** raw degrees (wrong metric), UTM (zone changes, more
dependencies), full geodesic for every operation (slower, unnecessary).

---

## D-003 — Group-aware cross-validation by trip

**Context:** observations from one trip are highly correlated. A random split
would put near-duplicate points in train and validation, inflating scores and
leaking trip-specific geometry.

**Decision:** all splits (model CV, Optuna, decoder tuning) group by trip; the 80
training trips are split with `GroupKFold(5)`, and the 10 student-test trips are
a separate development comparison set.

**Consequences:** honest generalisation estimates; fewer effective samples per
fold; the evaluator partition stays untouched.

**Alternatives rejected:** random row split (leakage), random trip split without
groups (leakage), tuning on student-test (optimistic and against the brief).

---

## D-004 — Ground truth isolated from prediction and UI

**Context:** the evaluator checks that ground truth stays server-side and that
the UI receives only the predicted route and results.

**Decision:** the prediction loader calls `load_trip(..., with_route=False)`;
`pathsnap/scoring.py` is the only module allowed to open `.route`, and it runs
after prediction. The API response contains aggregate metrics and predicted
geometry only. `tests/test_leakage.py` enforces this.

**Consequences:** a small amount of duplicated loading in scoring; an explicit
trust boundary that is easy to demonstrate. No reference-route overlay in the
live UI (report maps may show it because they are generated offline).

**Alternatives rejected:** passing the route to the predictor and ignoring it
(too easy to leak), scoring in the UI (ground truth on Laptop 2).

---

## D-005 — Python/scikit-learn stack; no deep learning, no XGBoost/LightGBM

**Context:** tabular candidate features, ~1.7M rows, CPU-only, Python 3.14, a
one-day build window.

**Decision:** NumPy/SciPy for geometry and sparse Dijkstra, scikit-learn for all
models (logistic regression, GaussianNB, decision tree, random forest, Extra
Trees, HistGradientBoosting, MLP, calibrated LinearSVC), Optuna TPE for
Bayesian optimisation, a custom NumPy HMM/Viterbi.

**Consequences:** fast, reproducible, no GPU, every model explains its
probabilities. HistGradientBoosting covers the boosting family. XGBoost/
LightGBM are excluded because of Python 3.14 wheel risk.

**Alternatives rejected:** PyTorch/TensorFlow (data size, calibration,
explainability), XGBoost/LightGBM (install risk), KNN (inference cost on 1.7M
rows).

---

## D-006 — Raw dataset not committed; artifacts and results are

**Context:** the provided dataset is 91 MB of course material; the repo must be
public and evaluator-accessible; GitHub limits and academic-integrity rules
apply.

**Decision:** git-ignore `data/`, `Student folder/` and
`PathSnap_student_materials/`. Commit source, configs, `results/` CSV/figures,
and the model artifact. Document placement in `data/README.md`.

**Consequences:** a fresh clone cannot train until the dataset is placed; the
README/status handoff gives the exact commands. Artifacts remain small.

**Alternatives rejected:** committing raw data (size, duplication), committing
only the model (not reproducible), external storage (extra dependency).

---

## D-007 — Append-only agent log + live status + ADRs

**Context:** four humans and multiple AI agents work concurrently. Chat history
is not shared, git diffs explain *what* but not *why* or *what is true now*, and
a single shared document becomes stale and conflict-prone.

**Decision:** use four layers:
1. `AGENTS.md` — stable read-first contract;
2. `docs/STATUS.md` — live state, claims, handoff (small edits);
3. `docs/AGENT_LOG.md` — append-only session history;
4. `docs/DECISIONS.md` — append-only ADRs.
Plus `docs/GLOSSARY.md` and `scripts/agent_status.py`.

**Consequences:** minimal merge conflicts (append at the bottom), explicit task
claims, reproducible handoffs. Cost: every session must spend a few minutes
updating the log/status; this is part of the Definition of Done.

**Alternatives rejected:** one giant `NOTES.md` (stale/conflicts), chat-only
coordination (not shared), GitHub Issues only (no local/offline record; still
useful as a complement).

---

## D-008 — Shared `pathsnap` library with thin API/UI

**Context:** training, scoring, decoding and serving must use identical code;
the UI must not import the model or data loaders.

**Decision:** all reusable logic lives in `pathsnap/`; `backend/` orchestrates
inference/scoring; `ui/` only calls HTTP. Scripts call the same library
functions. Contracts live in `pathsnap/contracts.py` and `config.py`.

**Consequences:** no train/serve skew; API tests can use the same pipeline;
owners can work in separate modules. Changes to shared contracts are announced
in the commit message and covered by tests; no approval gate.

**Alternatives rejected:** duplicating pipeline code in the API (skew), a
monolithic notebook (not deployable), UI importing the model (leaks ground
truth/clients).

---

## Template for a new ADR

```markdown
## D-0XX — Title

**Context:** what forces a decision now?
**Decision:** what exactly was chosen?
**Consequences:** what becomes easier/harder; what must others do?
**Alternatives rejected:** and why.
**Supersedes:** D-0YY (if any).
```
