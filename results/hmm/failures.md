# HMM failure analysis

Cases are selected mechanically from the classifier-vs-HMM table.

| Case | Trip | Classifier Top-1 | HMM Top-1 | Classifier continuity | HMM continuity | Classifier LCS-F1 | HMM LCS-F1 | Map |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| largest_top1_regression | 00000040 | 0.9538 | 0.9640 | 0.7770 | 0.9410 | 0.7759 | 0.9691 | `results/hmm/failure_maps/00000040_largest_top1_regression.html` |
| largest_continuity_recovery | 00000077 | 0.7976 | 0.8587 | 0.6527 | 0.8889 | 0.4394 | 0.9228 | `results/hmm/failure_maps/00000077_largest_continuity_recovery.html` |
| largest_sequence_regression | 00000009 | 0.8948 | 0.9237 | 0.7512 | 0.8654 | 0.8404 | 0.8925 | `results/hmm/failure_maps/00000009_largest_sequence_regression.html` |

## Interpretation checklist

- A Top-1 regression means smoothing selected a connected but wrong road; inspect junctions and parallel roads.
- A continuity recovery means the HMM removed an independent-classifier teleport.
- A sequence regression means continuity alone was not enough; inspect gaps, missing candidates, or over-smoothing.
