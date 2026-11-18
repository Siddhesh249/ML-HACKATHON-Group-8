# Report outline (mandatory structure)

The brief requires sections 2.2–7 in this order. Each row names the owner, the
evidence file and the figures/tables needed. Fill numbers only from `results/`.

| Section | Required content | Owner | Evidence | Figures/tables |
|---|---|---|---|---|
| Title page | Team, course, contributions, AI disclosure | A | — | — |
| 1. Introduction | Problem, two-stage approach, contributions | A | — | system diagram |
| 2.2 Dataset understanding & EDA | Preprocessing, patterns, features and why, missing values, GPS plots, network visualisations, ≥1 Folium map, partition labels | A | `results/eda/` | inventory table, sampling histogram, network stats, GPS/network map, Folium screenshot |
| 3.1 Model comparison | Consolidated table: model, family, config, CV macro F1, Top-1, continuity, student macro F1, log loss, inference time | B | `results/models/model_comparison.csv` | comparison table |
| 3.2 Per-model subsections | For every model: suitability, strengths/limitations, hyperparameters, loss, student-test performance, Top-1/macro F1/log loss, over/underfitting evidence | B | per-model folds/logs | small tables |
| 3.3 Model visualisations | CV macro F1 comparison, student Top-1 + macro F1, log loss, optional confusion matrices, real loss curves only | B | `results/figures/` | bar charts, confusion matrices, MLP/HistGB curves |
| 4. Bayesian optimisation | Search space, ranges, objective, trials, best values, tuned vs untuned on CV/Top-1/macro F1/log loss/inference time, cost | B | `results/optuna/` | trial history, parameter importance, tuned-vs-untuned table |
| 5. HMM & Viterbi | Hidden states, emission, transition, gaps/missing candidates/disconnected arcs, classifier-vs-HMM results and failures | C | `results/hmm/` | HMM diagram, comparison table, fixed-jump map |
| 6. Folium visualisation | GPS, network, reference, predicted route; colours/legend/layers/zoom; HTML committed | A | `results/maps/` | ≥1 screenshot in PDF |
| 7. API, UI and deployment | Endpoints, two-laptop architecture, screenshots of UI and API terminal logs, source must match demo | D | `results/screenshots/`, `docs/DEPLOYMENT.md` | UI screenshot, API log screenshot, architecture diagram |
| 8. Integrity / AI | AI disclosure, originality statement | all | — | — |
| Appendix | Reproduction commands, package versions, extra results | A | `README.md`, `docs/` | — |

## Per-model question template (compulsory)

For each model, answer:

1. Why is this algorithm suitable for PathSnap?
2. Major strengths and limitations for this task.
3. Important hyperparameters and initial values.
4. Loss function / optimisation objective.
5. Student-test performance.
6. Observation Top-1, macro F1, log loss (where applicable).
7. Overfitting or underfitting? What evidence?

Connect every answer to the actual experiment. No textbook-only descriptions.

## Figure checklist

- [ ] Cross-validation macro F1 comparison across all models.
- [ ] Student-test Top-1 and macro F1 comparison.
- [ ] Log loss comparison (where applicable).
- [ ] Confusion matrices (optional diagnostic).
- [ ] Real training/validation curves only for models with epoch histories.
- [ ] Model-selection comparison plot.
- [ ] Optuna history and parameter importance.
- [ ] Classifier-only vs HMM comparison.
- [ ] Folium map screenshot.
- [ ] API/UI screenshots.

## Numbers cross-check

Before submitting, every value in the PDF must appear in `results/` and in the
`viva.md` ledger. Run:

```bash
grep -n "TBD" viva.md
make test
```
