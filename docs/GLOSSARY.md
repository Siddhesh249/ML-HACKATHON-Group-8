# Glossary

Shared vocabulary. If a term changes meaning, add a new definition and say what
it supersedes. Metric definitions must match `pathsnap/metrics.py`.

| Term | Meaning |
|---|---|
| **Trip** | One vehicle trajectory and its local road network, stored in a folder named by the trip ID. |
| **Partition** | `train`, `student_test` or `evaluator_test`. Trips never cross partitions. |
| **Track** | The GPS observations of a trip: `(lon, lat, timestamp_seconds)`, nominally 1 Hz. |
| **Node** | A road-network coordinate. Its ID is the 0-based row index in `.nodes`. |
| **Arc** | A directed road segment between two nodes. Its ID is the 0-based row index in `.arcs`. |
| **Route** | The reference chronological sequence of arc IDs in `.route`. Not a per-observation label. Server-side only. |
| **Candidate** | One (observation, nearby arc) pair produced by candidate generation. |
| **Candidate set** | The top-K nearby arcs for one observation (default K=15, radius 60 m with fallbacks). |
| **Candidate miss rate** | Fraction of confident labelled observations whose reference arc is absent from the candidate set. A hard ceiling on Top-1. |
| **Positive** | A candidate whose arc equals the aligned reference arc. |
| **Alignment** | Mapping the reference route onto the track: project each GPS point onto the route polyline, enforce monotonic progress, assign the arc under the projection. |
| **Confidence flag** | Whether an observation's aligned label is reliable (projection distance ≤ threshold). Low-confidence observations are excluded from fitting. |
| **Coverage** | Fraction of observations with a confident label, or (in EDA) the fraction of route length covered. State which one you mean. |
| **Label** | The reference arc assigned to an observation by alignment. |
| **Feature** | One numeric value for a candidate, in the order defined by `config.FEATURE_NAMES`. |
| **Classifier** | The model that scores candidate features and returns a probability. |
| **Emission** | The HMM likelihood of an observation given a candidate arc; derived from the classifier probability and optionally a distance term. |
| **Transition** | The HMM likelihood of moving from one candidate arc to another, based on directed network distance versus GPS displacement. |
| **Hidden state** | The candidate arc assigned to an observation in the HMM. |
| **Viterbi** | Dynamic programming that finds the most probable state sequence in log space. |
| **Gap reset** | When the time between observations exceeds a threshold, the transition constraint is relaxed because the vehicle may have travelled far. |
| **Top-1 accuracy** | Per observation, the highest-probability candidate equals the reference arc. A missing truth counts as incorrect. |
| **Macro F1** | Unweighted mean of positive and negative F1 over candidate pairs. |
| **Positive F1** | F1 for the “is the reference arc” class. |
| **Log loss** | Binary cross-entropy of calibrated candidate probabilities. |
| **Route continuity** | Fraction of consecutive predicted arcs that are connected in the directed graph (strict), or within a small distance (relaxed). |
| **LCS F1** | Order-aware precision/recall/F1 computed from the longest common subsequence of deduplicated predicted and reference arc sequences. |
| **Arc-set F1** | Set overlap between predicted and reference arcs, ignoring order. |
| **Sequence exact match** | Predicted arc sequence equals the reference sequence after deduplication. |
| **Artifact** | The exported inference bundle: model, feature contract, HMM config, model card. |
| **Model version** | Short hash of the model name + hyperparameters, stored in the artifact. |
| **Evaluator partition** | The ten reserved trips used only for the final live evaluation. Never used for training, tuning or feature choice. |
| **Leakage** | Any use of evaluator data, reference labels, or trip-correlated splits that makes a reported result optimistic or invalid. |
| **Oracle** | A result computed using information the real system would not have at prediction time; not a valid model result. |
| **ADR** | Architecture Decision Record in `docs/DECISIONS.md`. |
| **Claim** | A temporary reservation in `docs/STATUS.md` of an area/file before editing it. |
| **Handoff** | The block in `docs/STATUS.md` that lets the next agent reproduce the current state. |
