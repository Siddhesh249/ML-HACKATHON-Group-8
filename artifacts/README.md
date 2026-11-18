# Artifacts

Committed inference assets. The API loads them at startup; it never trains or
reads `.route` during prediction.

## Files

| File | Contents |
|---|---|
| `model.joblib` | dict with the fitted estimator, feature contract, params, metrics, model version, git commit |
| `feature_config.json` | feature names/version, smoothing window, candidate radius/top-K, label threshold, graph-feature flag |
| `hmm_config.json` | `beta_m`, `sigma_m`, `lam`, `max_speed_mps`, `gap_reset_s`, `uturn_penalty`, `transition_slack_m` |
| `model_card.json` | human-readable model card: data partition, rows, positives/negatives, metrics, environment |
| `README.md` | this file |

## Export

```bash
python scripts/06_export_artifacts.py \
  --dataset cache/dataset.npz \
  --selected results/models/selected_model.json \
  --best-params results/optuna/best_params.json \
  --artifact-dir artifacts
```

The exporter:

1. reads the selected model and tuned parameters;
2. refits on all training rows (negative-subsampled);
3. writes the artifact + feature/HMM/model-card configs;
4. records the git commit and environment.

## Contract

The API refuses to load an artifact whose `feature_names` differ from
`pathsnap/config.py`. If you change the feature list, bump `FEATURE_VERSION`,
rebuild the dataset, retrain, and re-export.

## Size

The model file should stay small enough for GitHub (well under 100 MB). Do not
commit the dataset, cache or OOF probability arrays.

## Verify

```bash
python - <<'PY'
from pathsnap import models
a = models.load_artifact("artifacts/model.joblib")
print(a["model_name"], a["model_version"], len(a["feature_names"]))
print(a["metrics"])
PY
```
