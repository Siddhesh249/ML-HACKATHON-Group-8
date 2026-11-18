# Agent log — append-only lab notebook

> **Newest entries at the bottom.** Append one entry per logical session.
> Never edit a previous entry except a typo. Every result needs a path.
> Read the last three entries at the start of a session.
>
> Entry ID format: `AL-YYYYMMDD-HHMM-<agent>`.

---

## AL-20261118-0830-D — scaffold, contracts, tests

- **Task:** T00–T09 (Phase 0 scaffold)
- **Agent:** D (platform/integration), with AI coding assistance
- **Branch/commit:** `main` (initial scaffold)
- **Files:** `README.md`, `viva.md`, `AGENTS.md`, `pathsnap/`, `backend/`, `ui/`, `scripts/`, `tests/`, `docs/`, `requirements.txt`, `Makefile`
- **Commands:** `make smoke`, `python3 -m pytest -q`
- **Result:** 27 tests pass (geo, IO, candidates, labels, metrics, HMM, API, leakage). FastAPI `/health`, `/trips` and `/predict` contract verified with an injected predictor. No live artifact yet.
- **Interpretation:** the vertical slice is wired end to end at the code level; the missing link is a trained/exported model.
- **Decision/ADR:** D-001…D-008
- **Next:** build the dataset, train a baseline, export, run a live prediction.
- **Blockers:** none.

---

## AL-20261118-1020-D — EDA smoke + full dataset cache

- **Task:** T10 preparation, T13/T14 baseline
- **Agent:** D
- **Branch/commit:** `main` (working tree)
- **Files:** `scripts/01_eda.py`, `scripts/02_build_dataset.py`, `pathsnap/dataset.py`, `cache/` (ignored)
- **Commands:**
  - `python3 scripts/01_eda.py --data-root ./data --max-trips 5`
  - `python3 scripts/02_build_dataset.py --data-root ./data --out cache/dataset.npz`
- **Result:**
  - 5-trip EDA sample: label coverage 93.1–98.4%.
  - Full cache: 80 train trips, 208,417 observations, 1,732,459 candidate rows, 18 features, 158,563 positives, 1,184,963 negatives, 388,933 ignored rows. Build time ~59 s.
  - Per-trip candidate miss rates are recorded in `cache/dataset.json` stats (not yet aggregated in `results/`).
- **Interpretation:** candidate generation, alignment and graph features scale to all 80 trips on this machine; the dataset is ready for model comparison.
- **Decision/ADR:** D-003 (GroupKFold) enforced in the builder; no evaluator data used.
- **Next:** run the full 5-fold model comparison (`make train`), select, export, live smoke.
- **Blockers:** none.

---

## AL-20261118-1035-D — cross-agent communication protocol

- **Task:** documentation/process (not in the original board; supports all tasks)
- **Agent:** D
- **Branch/commit:** `main` (working tree)
- **Files:** `AGENTS.md`, `docs/STATUS.md`, `docs/AGENT_LOG.md`, `docs/DECISIONS.md`, `docs/GLOSSARY.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `scripts/agent_status.py`, `Makefile`, `README.md`
- **Commands:** `make status`
- **Result:** four-layer protocol: read-first contract (`AGENTS.md`), live state (`STATUS.md`), append-only history (`AGENT_LOG.md`), immutable rationale (`DECISIONS.md`), plus a shared glossary and a status snapshot tool.
- **Interpretation:** agents can now reconstruct *what is true*, *what changed*, *why*, and *what to do next* without reading a long chat history. Append-only files minimise merge conflicts between concurrent agents.
- **Decision/ADR:** D-007
- **Next:** every task's Definition of Done now includes log + status + viva updates.
- **Blockers:** none.

---

## AL-20261118-1105-D — label alignment correction: stop gating labels on monotonicity

- **Task:** T12 (baseline label policy), data-quality investigation
- **Agent:** D
- **Branch/commit:** `main` (working tree)
- **Files:** `pathsnap/labels.py`, `scripts/02_build_dataset.py`, `pathsnap/dataset.py`, `results/labels/`, `docs/DATA_NOTES.md`
- **Commands:**
  - `python3 scripts/02_build_dataset.py --data-root ./data --out cache/dataset.npz --force`
  - diagnostic plots of GPS vs reference route for trips 11, 33, 54, 69
- **Finding:** the “low coverage” on 13 trips was mostly an artifact of enforcing monotonic route progress. Trips 11/69 are loop/out-and-back or reversed routes: the nearest route arc is only 0–14 m away, but the monotonic arclength jumps, making the assigned segment kilometres away. The reference geometry is fine; the monotonic gate was wrong.
- **Fix:** labels use the nearest route segment/arc; `progress_m` and `backward_fraction` remain diagnostics. Confidence is still capped by a 30 m projection distance.
- **Result:**
  - coverage: 81.66% → **96.80% mean** (median 99.87%)
  - labelled observations: 158,760 → **183,052 / 208,417**
  - positives: 158,563 → **182,895**
  - candidate miss rate: **0.17% mean / 0.086% weighted**, worst trip 7.44%
  - only train `00000033` and `00000054` remain genuinely unreliable (and student-test `00000077`)
  - sources: `results/labels/summary.json`, `results/labels/coverage.csv`
- **Interpretation:** labels are now robust to route direction and loops; the two remaining trips are real reference-route failures, not an algorithm artifact.
- **Decision/ADR:** D-009
- **Next:** rebuild features (done), rerun model comparison on the corrected dataset.
- **Blockers:** none.

---

## AL-20261118-1125-D — ownership correction: stop training, remove exploratory results

- **Task:** process correction (no task ID)
- **Agent:** D
- **Files removed:** `cache/`, `results/{eda,labels,models,models_small,logs,optuna,hmm,integration,maps,figures,screenshots}/`, temporary artifacts
- **Why:** D started model training and label work to unblock the live API. That
  is B's and A's ownership. The correct integration approach is a stub/fixture
  until B exports an artifact.
- **Action:** killed the training run; removed every exploratory dataset, model,
  OOD and result output; kept only the scaffold, interfaces, tests and docs.
- **Current state:** repository contains scaffold + assignments only. A/B/C own
  the experiment work; D owns backend/UI/deployment/integration.
- **Next:** commit the scaffold; A/B/C start from `docs/ASSIGNMENTS.md`.
- **Blockers:** `/predict` live is intentionally blocked until B exports
  `artifacts/model.joblib`.

---

<!-- Append new entries below this line. Do not edit above it. -->

## AL-20261118-1112-B — Phase 1: dataset, metrics harness, 8-model comparison, ablation, preliminary artifact

- **Task:** T13, T14, T15, T16 (+ preliminary artifact for the Phase-1 gate)
- **Agent:** B (B-owned files only)
- **Branch/commit:** `feat/b-models` from `main` @ `b41ffea`
- **Files:** `pathsnap/dataset.py` (`top1_by_mask`, `cv_split_table`, `write_dataset_summary`), `pathsnap/features.py` (`FEATURE_GROUPS`, `feature_indices`; feature computation unchanged), `pathsnap/models.py` (`fold_scorer` hook, `repo_relative`), `scripts/02_build_dataset.py` (`--summary-only`), `scripts/03_train_models.py` (per-fold val/train metrics, baseline row, 8 families), `scripts/03b_feature_ablation.py` (new), `scripts/06_export_artifacts.py` (relative paths), `tests/test_models.py` (new, 5 tests), `results/models/*`, `artifacts/{model.joblib,feature_config.json,model_card.json,hmm_config.json}`
- **Commands:**
  - `python3 scripts/02_build_dataset.py --data-root ./data` (then `--summary-only`)
  - `python3 scripts/03_train_models.py --folds 5 --profile full --out-dir results/models` → log `results/logs/03_train_models_full.log`
  - `OMP_NUM_THREADS=8 python3 scripts/03b_feature_ablation.py --model hist_gradient_boosting --folds 5 --repeats 3` → log `results/logs/03b_feature_ablation.log`
  - `python3 scripts/06_export_artifacts.py --dataset cache/dataset.npz --selected results/models/selected_model.json --artifact-dir artifacts`
- **Result:**
  - Dataset: 80 train trips, 208,417 obs, 1,732,459 candidate rows (182,884 pos / 1,402,460 neg / 147,115 ignored); all split checks true (no trip in two folds, train only, no NaN, feature order = config) — `results/models/dataset_summary.json`, `cv_splits.csv`, `dataset_per_trip.csv`
  - Comparison (CV observation Top-1 / macro F1 / log loss): HGB 0.887/0.928/0.084, RF 0.886/0.923/0.098, ET 0.885/0.917/0.115, DT 0.879/0.898/0.142, MLP 0.878/0.922/0.096, LinearSVM 0.871/0.898/0.123, LR 0.871/0.857/0.179, GNB 0.845/0.137/4.746, nearest-road baseline 0.575 — `results/models/model_comparison.csv`, per-fold `results/models/*_folds.json`
  - Selected: HistGradientBoosting (preliminary, untuned) — `results/models/selected_model.json`
  - Ablation: distance only 0.567 → local 0.879 → +relative 0.880 → full 0.887; full − heading 0.646; full − graph 0.884 — `results/models/feature_ablation.csv`, `feature_ablation.png`
  - Permutation importance: abs_heading_diff 0.193, log_distance 0.146, projection_ratio 0.133, distance_ratio_best 0.071 — `results/models/feature_importance.csv`, `feature_importance.png`
  - Artifact `artifacts/model.joblib` v`6c229c610200` (1.1 MB, fit on 731,586 rows). `backend.service.Predictor` loads it (ready=true); one live prediction on student-test `00000000`: classifier point Top-1 0.957 / HMM 0.950, strict continuity 0.742 → 0.837. Sanity check only — not used for selection or tuning.
  - `make test` 32 passed; `make smoke` OK.
- **Interpretation:** learned models beat nearest-road by ~0.31 Top-1; heading is the key feature. The top three models are within fold noise (std ≈ 0.026); HGB wins on the guardrails (calibration, F1) and fit time, and is less overfit than RF (train-trip Top-1 0.923 vs 0.939). Graph/temporal features add little per point; continuity is the HMM's job.
- **Notes / caveats:** MLP and LinearSVM fit times were measured while the ablation job shared the CPU (upper bounds). Committed JSON uses repo-relative paths (no machine paths). `artifacts/hmm_config.json` contains `config.py` defaults written by the exporter — C owns it and should overwrite with tuned values. The dataset build also rewrote A's `results/labels/` and `results/eda/candidate_miss_summary.csv`; B does not commit those (A owns them).
- **Decision/ADR:** none (selection rule, splits and feature contract follow EXPERIMENTS.md / D-003 unchanged; FEATURE_VERSION stays v1).
- **Next:** B: Optuna on HGB (T23), tuned-vs-untuned on CV + student-test (T24), report §3–4. D: integrate the artifact (T19/T20). C: classifier-vs-HMM with this artifact (T26).
- **Blockers:** none for B. `httpx` is missing from `requirements.txt` (needed by `tests/test_api.py`) — D to add.

---

## AL-20261118-1130-A — Phase 1: full EDA, candidate sweep, label coverage, first Folium maps

- **Task:** T10, T11, T12, T21
- **Agent:** A (A-owned files only)
- **Branch/commit:** `feat/a-data` from `main` @ `b41ffea`
- **Files:** `scripts/01_eda.py` (rewritten: raw file checks, quality checks, flags, network stats, 9 figures; fixed the representative-trip plot which indexed arcs by arc rows), `scripts/01b_candidate_sweep.py` (new), `scripts/01c_label_coverage.py` (new), `pathsnap/labels.py` (`project_onto_route`, `route_order_diagnostics`; label output unchanged), `pathsnap/visualize.py` (`show_reference`, legend, fit bounds), `visualisation/make_maps.py` (new), `tests/test_labels.py` (+1 test), `docs/DATA_NOTES.md`, `results/eda/*`, `results/labels/*`, `results/maps/*`, `results/figures/map_*.png`
- **Commands:** `python3 scripts/01_eda.py --data-root ./data`; `python3 scripts/01b_candidate_sweep.py --data-root ./data`; `python3 scripts/01c_label_coverage.py --data-root ./data`; `python3 visualisation/make_maps.py --data-root ./data`; `make test` (28 passed), `make smoke`
- **Result:**
  - Inventory: 90/90 trips complete; 208,417 train / 19,160 student-test points; 0 invalid GPS rows, 0 repeated/backward timestamps; 96.7% of steps are 1 s; 30 trips with a gap > 30 s, max 153,064 s (train `00000090`); 966 raw steps > 55 m/s in 28 trips — `results/eda/summary.json`, `quality_checks.csv`, `flags.csv`
  - Network: median arc 25.6 m (max 9,308 m); 106 zero-length, 24 self-loop, 1,903 duplicate arcs; median 19 weak components per trip — `results/eda/network_stats.csv`
  - Routes: 11/90 have 1–3 breaks (15–597 m) — corrects the earlier "always connected" note; 8 loop/out-and-back (time-vs-progress rank corr < 0.8) — `results/eda/inventory.csv`
  - Candidates: top-K drives the miss rate (K=5 ≈1.2% → K=15 ≈0.09% → K=30 ≈0.002%); radius ≥ 30 m is flat on labelled points; production 60 m + fallbacks, K=15: 0.092% train / 0.032% student-test, 8.7 candidates/obs — `results/eda/candidate_sweep.csv`, `candidate_sweep_choice.json`
  - Labels: train 183,052/208,417 labelled (mean coverage 96.8%, median 99.87%); student-test 19,018/19,160 (min 98.0%); unreliable: train `00000033` (vehicle parked ~120 m off-network), `00000054` (steady 100–200 m offset) — `results/labels/coverage.csv`, `summary.json`, `visual_check_*.png`
  - Maps: `results/maps/train_00000046_reference.html`, `train_00000033_reference.html` + screenshots `results/figures/map_*.png`
- **Interpretation:** the data is clean at the row level; the difficulties are structural (gaps, GPS jumps, fragmented networks, contracted arcs) and in the reference routes (breaks, loops, two offset trips). Candidate generation is not a bottleneck; Top-1 errors will come from ranking, not recall. Label numbers match B's dataset build exactly (183,052 labelled).
- **Decision/ADR:** none — candidate config and label policy (D-009) unchanged. The `backward_fraction` diagnostic is superseded for flagging by `route_order_corr` (diagnostic only).
- **Correction:** the earlier note that student-test `00000077` is unreliable came from the superseded monotonic alignment; with D-009 its coverage is ≥ 98%.
- **Next:** B to stop `02_build_dataset.py` writing into `results/labels/`; A Phase 2: T28 prediction maps, T29 report §2; optional check of opposite-direction labels on out-and-back trips.
- **Blockers:** none.

---

## AL-20261118-1210-C — Phase 1 HMM transitions and verification

- **Task:** T17, T18 (Phase 1 C package)
- **Agent:** C, with AI coding assistance
- **Branch/commit:** `feat/c-hmm` @ `7a9dab0` (rebased onto current `main` during conflict resolution)
- **Files:** `pathsnap/hmm.py`, `docs/HMM_DESIGN.md`, `tests/test_hmm.py`, `scripts/benchmark_hmm.py`, `README.md`, `docs/STATUS.md`, `viva.md`
- **Commands:** `python -m pytest tests/test_hmm.py -q --basetemp .pytest-tmp`; `python -m compileall -q pathsnap scripts/benchmark_hmm.py`; `git diff --check`
- **Result:** 10 focused HMM tests pass; package and benchmark script compile. Added coverage for same-arc progress, reversed progress, unreachable targets, gap reset, U-turn penalty and empty candidate layers. Real-trip latency CSV was not produced because the workspace has no raw trip data.
- **Interpretation:** the decoder’s corridor/Dijkstra transition contract is executable and documented; the remaining evidence boundary is a real mounted trip and B’s review.
- **Decision/ADR:** no new ADR; implementation follows D-001 and D-003.
- **Next:** B reviews `docs/HMM_DESIGN.md`; run `python -m scripts.benchmark_hmm` on the largest permitted training trip; then proceed to T25/T26 after the artifact.
- **Blockers:** `make` is unavailable; full pytest collection needs the missing `httpx` dependency; raw dataset is not mounted for the required latency record.

---

## AL-20261118-1245-D — live API/UI integration on B’s preliminary artifact

- **Task:** T19, T20, T31, T32; T33 local loopback
- **Agent:** D, with AI coding assistance
- **Branch/commit:** `feat/d-integration`
- **Files:** `ui/app.py` (predicted-route layer + legend), `scripts/run_eval.py` (stage timings), `requirements.txt` (`httpx`), `results/integration/*`, `results/screenshots/*`, `report/07_deployment.md`, `docs/STATUS.md`, `README.md`, `viva.md`
- **Commands:**
  - `make test`
  - `uvicorn backend.app:app --host 127.0.0.1 --port 8000`
  - `curl /health`, `/trips`, `POST /predict` (00000052, 00000056 × hmm/classifier)
  - `streamlit run ui/app.py` + Playwright screenshot
  - `python scripts/run_eval.py --partition student_test --modes hmm classifier`
- **Result:**
  - 39 tests pass.
  - 10/10 student-test trips pass in both modes; 20 rows in `results/integration/run_log.csv`.
  - Classifier: Top-1 0.897 mean, continuity 0.756, LCS F1 0.763, p50 438 ms / p95 1179 ms.
  - HMM (defaults): Top-1 0.851, continuity 0.839, LCS F1 0.729, p50 1213 ms / p95 3579 ms.
  - HMM improves continuity on 10/10 trips but lowers Top-1 on 9/10 → C tuning required before freezing.
  - Live API worst case: 4.07 s on 3,958 points (`00000056`, HMM); classifier 1.32 s.
  - Evidence: `results/integration/summary.json`, `stage_timings.csv`, `per_trip_comparison.csv`, `results/screenshots/`, `report/07_deployment.md`.
- **Interpretation:** the two-process platform is live and server-scored; the remaining integration risk is the default decoder configuration, not the API/UI plumbing.
- **Decision/ADR:** no new ADR; uses D-001/D-004. Added `httpx` for the Starlette test client (D-005 dependency set).
- **Next:** C tunes the HMM and publishes `results/hmm/classifier_vs_hmm.csv`; B exports the tuned artifact; D rehearses two laptops and refreshes §7.4–7.5.
- **Blockers:** evaluator trips not mounted; physical second laptop not available in this session.

---

## AL-20261118-1310-D — simplify viva.md and the docs maintenance protocol

- **Task:** T02 (viva), cross-agent docs (T37)
- **Agent:** D, with AI coding assistance
- **Branch/commit:** `docs/simplify-viva`
- **Files:** `viva.md` (rewritten: 1000 → 350 lines), `AGENTS.md`, `README.md` (§14, §16), `docs/ASSIGNMENTS.md`, `docs/DATA_NOTES.md`, `scripts/agent_status.py`, `docs/STATUS.md`
- **Commands:** `python3 -m pytest -q`; `python3 scripts/agent_status.py`; `grep -n TBD viva.md`
- **Result:** `viva.md` is now 14 short sections: task, data, preprocessing, features, models, HMM, metrics, API/UI, numbers ledger, Q&A bank (rapid-fire one-liners + common questions + per-owner minimum), demo script, checklist, AI disclosure, maintenance rules. Numbers are unchanged and still point at `results/`. `AGENTS.md`, README §16 and `scripts/agent_status.py` now ask for narrow edits (ledger row / one paragraph / one Q&A row) instead of a full-document update.
- **Interpretation:** the viva document is cheap to read and cheap to keep true; maintenance is a per-fact edit rather than a per-task rewrite.
- **Decision/ADR:** none; documentation scope only.
- **Next:** keep the ledger current as B/C publish Optuna and HMM tuning; add unanswered viva questions to `viva.md` §10 as they appear.
- **Blockers:** none.

---

## AL-20261118-1345-D — Phase 1 close-out: T17/T18 evidence, model-choice table, cleanup

- **Task:** T17, T18 (last open Phase-1 rows); T02 (viva)
- **Agent:** D, with AI coding assistance (cross-cover where the owner was unavailable)
- **Branch/commit:** `chore/phase1-closeout`
- **Files:** `docs/HMM_DESIGN.md`, `results/hmm/transition_benchmark.csv`, `viva.md` (§5 model table + reasoning, §6 latency, §9 ledger), `README.md` (T17/T18, milestones), `docs/STATUS.md`, `AGENTS.md`, `.gitignore`, `.env.example`
- **Commands:**
  - `python3 -m scripts.benchmark_hmm --data-root ./data --partition train --trip-id 00000054` (and `00000084`)
  - `python3 -m pytest -q`
  - `python3 scripts/agent_status.py`
- **Result:**
  - T17: code-vs-doc review of `docs/HMM_DESIGN.md` against `pathsnap/hmm.py`, `network.py`, `config.py`. Two implementation details were undocumented (300 m displacement floor, 5 km Dijkstra bound); both added. No code change needed.
  - T18: `results/hmm/transition_benchmark.csv` — 5.17 s warm transitions on 19,313 observations (0.268 ms/transition) and 4.62 s on 16,031 observations (0.288 ms), defaults.
  - `viva.md` §5 now presents the model-comparison table as the selection evidence: fixed rule (pooled OOF Top-1, tie-break macro F1 then log loss), why HGB won, what the other families show, and the honest note that the RF gap is inside fold noise.
  - Phase 1 rows T10–T21 are all ✅; repo junk removed (caches, runtime logs), `.env.example` un-ignored.
- **Interpretation:** Phase 1 is closed with every gate item backed by a file in `results/`; Phase 2 owns tuning, report and the physical rehearsal.
- **Decision/ADR:** none; the model-selection rule was already implied by `selected_model.json` and is now stated explicitly.
- **Next:** Phase 2 — B's Optuna (T23/T24), C's decoder tuning (T25–T27), A's report/maps (T28/T29), D's two-laptop rehearsal (T33) and final hardening (T36).
- **Blockers:** none for Phase 1.

---

## AL-20261118-1420-D — dark-theme UI restyle with local preview

- **Task:** T20, T31 (UI); unblocked polish while B/C tuning is pending
- **Agent:** D, with AI coding assistance
- **Branch/commit:** `feat/d-ui-polish`
- **Files:** `ui/app.py`, `.streamlit/config.toml`, `results/screenshots/ui_dark_*.png`, `report/07_deployment.md`, `docs/STATUS.md`, `README.md`
- **Commands:** `python3 -m uvicorn backend.app:app --port 8000`; `PATHSNAP_API_URL=http://127.0.0.1:8000 python3 -m streamlit run ui/app.py --server.port 8501`; Playwright screenshots; `python3 -m pytest -q`
- **Result:** dark theme (config + CSS), branded header with `Ready` / `student-test fallback` / model pills, three-step layout, metric cards with one-line hints, metadata chips, dark Esri basemap with dark Leaflet controls, explicit fallback banner, stale-result notice, retry on failure, `st.iframe` replaces deprecated `components.html`, download button now emits real GeoJSON. Screenshots: `ui_dark_hmm_trip00000000.png`, `ui_dark_classifier_trip00000000.png`, `ui_dark_empty_state.png`. 39 tests pass.
- **Interpretation:** the client now looks like a product for the evaluator without changing the API contract or the ground-truth boundary.
- **Decision/ADR:** none; presentation only. Theme is dark by team request (projector contrast to be checked during the physical rehearsal).
- **Next:** collect review comments; then two-laptop rehearsal and screenshot refresh after tuned artifacts.
- **Blockers:** none.

---

## AL-20261118-1430-B — Phase 2: Optuna (T23), tuned vs untuned (T24), student-test comparison (T22), report §3–4 (T30)

- **Task:** T22, T23, T24, T30
- **Agent:** B
- **Branch/commit:** `feat/b-phase2` from `main` @ `9d2655f`
- **Files:** `scripts/04_tune.py` (5-fold default, untuned default enqueued as trial 0, per-trial metrics, cost, accepts `--data-root` so `make tune` works), `scripts/04b_student_test_eval.py` (new), `scripts/04c_model_figures.py` (new), `scripts/06_export_artifacts.py` (applies Optuna params only if the T24 decision says so), `scripts/02_build_dataset.py` (no longer writes A's `results/labels/`, `results/eda/`), `results/optuna/*`, `results/models/{student_test_comparison,cv_vs_student_test,student_test_per_trip,tuned_vs_untuned}.csv`, `results/models/{tuned_vs_untuned_decision,learning_curves}.json`, `results/figures/{model_comparison_bars,learning_curves}.png`, `report/03_models.md`, `report/04_optuna.md`
- **Commands:**
  - `python3 scripts/02_build_dataset.py --data-root ./data --partitions student_test --out cache/student_test.npz`
  - `python3 scripts/04_tune.py --trials 40 --folds 5` (34.9 min)
  - `python3 scripts/04b_student_test_eval.py --data-root ./data`
  - `python3 scripts/04c_model_figures.py`
  - `python3 scripts/06_export_artifacts.py --artifact-dir <scratch>` → version `6c229c610200` (identical to the committed artifact)
- **Result:**
  - Optuna: 40/40 trials complete, 0 failed; CV Top-1 range 0.8843–0.8872; best trial 18 (lr 0.0223, 592 trees, 19 leaves, min leaf 97, L2 3.69) 0.8872 vs untuned 0.8868; fANOVA importance `max_leaf_nodes` 0.76 — `results/optuna/best_params.json`, `param_importances.json`, `trials.csv`, `study.db`
  - Tuned vs untuned: CV +0.0004 Top-1, positive F1 −0.0014, log loss +0.0009; student-test −0.0043 Top-1 → **keep untuned** — `results/models/tuned_vs_untuned.csv`, `tuned_vs_untuned_decision.json`
  - Student-test (8 models, refit on 80 train trips): HGB first on Top-1 0.882, macro F1 0.925, log loss 0.078; top-2 and bottom-3 ranks identical to CV — `results/models/student_test_comparison.csv`, `cv_vs_student_test.csv`
  - Learning curves (real histories, HGB + MLP only) — `results/figures/learning_curves.png`, `results/models/learning_curves.json`
- **Interpretation:** the default HGB configuration already sits on the Top-1 plateau; the residual ≈ 11% error is per-observation ambiguity that the HMM, not boosting hyperparameters, must resolve. The artifact is unchanged, so C's and D's numbers on `6c229c610200` stay valid.
- **Decision/ADR:** T24 rule = `docs/EXPERIMENTS.md` §3 (CV Top-1 with positive-F1/log-loss guardrails, CV only). An earlier draft rule in the evaluation script ("log loss ≤ 5% worse") was replaced by the documented §3 rule after the CV numbers were seen; under either rule student-test would not have been used. No ADR (selection policy unchanged).
- **Next:** C: decoder tuning T25 on artifact `6c229c610200` (see the T17 review follow-ups). B: none blocking; optional richer sequence features if time allows.
- **Blockers:** none.

---

## AL-20261118-1440-B — T17 B review of the HMM (formulas match; same-arc jitter finding)

- **Task:** T17 (B review; complements D's cross-review in `docs/HMM_DESIGN.md` §11)
- **Agent:** B
- **Branch/commit:** `feat/b-phase2` (review originally prepared at 12:15 on a separate branch against `3bb24a7`; ported here to avoid conflicts)
- **Files:** `docs/reviews/T17_hmm_review_B.md`, `scripts/review_hmm_same_arc.py`, `results/models/hmm_review_same_arc.{csv,json}`, `README.md` (T17 note), `docs/STATUS.md` (C next action), `viva.md` (§HMM known limitation)
- **Commands:** `python3 scripts/review_hmm_same_arc.py --data-root ./data`
- **Result:** design-note formulas match `pathsnap/hmm.py`; classifier → HMM interface correct. Finding 1: the same-arc rule forbids 18.8% of 148,039 true same-arc steps on 80 train trips (median backward 3.4 m, p95 14.3 m; 0.85% beyond 15 m) — `results/models/hmm_review_same_arc.json`. Findings 2–4: Dijkstra cap must scale with a tuned gap reset; include λ = 0; document the uniform start prior.
- **Interpretation:** plausible contributor to the default HMM's Top-1 loss (`results/integration/per_trip_comparison.csv`); hypothesis for C to test in T25.
- **Decision/ADR:** none.
- **Next:** C applies the follow-ups in T25.
- **Blockers:** none.

---

## AL-20261118-1505-D — UI header fix + viva Phase 1/Phase 2 journey

- **Task:** T20/T31 (UI), T02 (viva), T36 partial hardening
- **Agent:** D, with AI coding assistance
- **Branch/commit:** `feat/d-ui-fix` + `docs/d-journey-viva`
- **Files:** `ui/app.py`, `viva.md` (rewritten as a journey), `README.md` (§14/§16 refs), `AGENTS.md` (ledger ref), `report/07_deployment.md`, `docs/STATUS.md`, `results/screenshots/ui_dark_*.png`
- **Commands:** local API + Streamlit; Playwright screenshots; `python3 -m pytest -q`; secrets/data audit
- **Result:**
  - UI: `stHeader` was fixed-position and clipping the header card; hidden it and used container padding. Removed the ground-truth sidebar card (claim stays in the caption, report and leakage tests). Stopped caching failed `/health`/`/trips` probes so the UI recovers as soon as the API returns.
  - Viva: restructured into Part I (Phase 1 build journey), Part II (Phase 2 decisions: Optuna → keep untuned, student-test, decoder suspects, deployment) and Part III (metrics, ledger, Q&A bank, demo, checklist). Rapid-fire questions kept.
  - Report §7: records the model-freeze decision; only the HMM columns can change.
- **Interpretation:** the viva now reads as a narrative of decisions rather than a flat reference, and the two phases are visibly separate.
- **Decision/ADR:** none; documentation and presentation changes only.
- **Next:** C's decoder tuning (T25–T27), physical two-laptop rehearsal, final screenshots.
- **Blockers:** decoder results and second laptop are still external.

---

## AL-20261118-1520-D — error-state evidence and hardening spot-check

- **Task:** T32 (errors handled), T36 partial
- **Agent:** D, with AI coding assistance
- **Branch/commit:** `feat/d-error-states`
- **Files:** `results/screenshots/ui_dark_api_offline.png`, `report/07_deployment.md`, `docs/STATUS.md`, `README.md`, `viva.md`
- **Commands:** UI started with `PATHSNAP_API_URL=http://127.0.0.1:9999`; Playwright screenshot; `git ls-files` data audit; secrets grep; `gh repo view`; logged-out `curl`
- **Result:**
  - Offline state verified: red `API unreachable` badge + `API offline` pill + the exact Laptop-1 start command, captured as `ui_dark_api_offline.png`.
  - Hardening: only `data/README.md` is tracked from the data tree; no secrets in the repo; repo public and reachable logged-out (HTTP 200); largest tracked file 5.9 MB (A's map HTML).
  - README T32 now cites the error-state screenshot; report §7 lists all four UI evidence images.
- **Interpretation:** "errors handled" is now demonstrated, not just implemented; the remaining hardening gate is the final tag after decoder tuning and the physical rehearsal.
- **Decision/ADR:** none.
- **Next:** C's decoder results, then refresh the HMM columns; physical two-laptop rehearsal; `v1.0-final`.
- **Blockers:** decoder results and the second laptop are external.

---

## AL-20261118-1510-A — Phase 2: classifier-vs-HMM maps (T28) and claim/evidence audit (T34)

- **Task:** T28, T34 (T29 report handed to another member; T35 needs the people)
- **Agent:** A
- **Branch/commit:** `feat/a-phase2` from `main` @ `17c19bd`
- **Files:** `pathsnap/visualize.py` (multiple prediction layers, zoom bounds, basemap option, stacked widths), `visualisation/make_maps.py` (`--predict`: API-identical pipeline, server-side scoring, full + zoom screenshots, failure metric), `scripts/audit_claims.py` (new), `tests/test_audit.py` (new), `results/maps/{student_test_*_classifier_vs_hmm.html,prediction_maps.csv}`, `results/figures/map_student_test_*`, `results/audit/*`, `README.md` (stale claims table fixed), `docs/STATUS.md`, `viva.md` (ledger row)
- **Commands:** `python3 visualisation/make_maps.py --data-root ./data --predict`; `python3 scripts/audit_claims.py`; `make test` (41 passed)
- **Result:**
  - Maps (artifact `6c229c610200`, default HMM config): `00000052` classifier 0.969 → HMM 0.992 (the HMM keeps the ramp where the classifier jumps onto the motorway); `00000056` 0.818 → 0.781, continuity 0.74 → 0.86; `00000077` 0.798 → 0.639 — matches D's API run within 0.003 — `results/maps/prediction_maps.csv`
  - Failure analysis: on `00000077`, 77.4% of the 860 points the HMM gets wrong but the classifier gets right are the **opposite-direction twin** of the correct arc (16.7% on `00000056`); consistent with B's T17 finding that backward jitter makes "stay on this arc" infeasible
  - Audit: 26 ledger rows → 20 verified automatically + 1 manually, 4 TBD (C), 1 discrepancy (D: 4.07 s quoted vs 3.86 s in the cited `run_log.csv`); 0 broken path references — `results/audit/README.md`, `claims_audit.csv`, `summary.json`
- **Interpretation:** the HMM's Top-1 loss is dominated, on the worst trip, by direction flips that are invisible on a map; this is the most actionable target for C's T25.
- **Decision/ADR:** none.
- **Next:** C: T25 with a direction-aware fix; D: resolve the latency citation; re-run `scripts/audit_claims.py` and `make_maps.py --predict` after T25.
- **Blockers:** T34 cannot be "clean" until C's TBD rows exist.

---
