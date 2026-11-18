# Data

The provided dataset is **not committed** (91 MB of course material).

## Expected layout

```
data/
├── train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
├── student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}
└── evaluator_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}
```

- `train`: 80 trips, for fitting and cross-validation.
- `student_test`: 10 trips, for development-stage comparison.
- `evaluator_test`: 10 reserved trips, placed on Laptop 1 at evaluation time.

The student package contains only `train/` and `student_test/`. The original
workspace folders `Student folder/` and `PathSnap_student_materials/` are
git-ignored. For local runs you can symlink them:

```bash
mkdir -p data
ln -s "../Student folder/train" data/train
ln -s "../Student folder/student_test" data/student_test
# evaluator trips are provided separately
```

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `PATHSNAP_DATA_ROOT` | `./data` | Root containing partition folders |
| `PATHSNAP_EVALUATION_PARTITION` | `evaluator_test` | Partition the UI lists |
| `PATHSNAP_FALLBACK_PARTITION` | `student_test` | Local demo fallback |

## Do not commit

- raw `.track`, `.nodes`, `.arcs`, `.route` files;
- caches (`cache/`);
- ground-truth copies anywhere outside Laptop 1;
- credentials or API keys.

## Verify

```bash
python - <<'PY'
from pathsnap import io
for part in ("train", "student_test"):
    print(part, len(io.discover_trips("data", partition=part)))
PY
```
