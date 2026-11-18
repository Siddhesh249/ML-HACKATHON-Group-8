# HMM failure analysis

Cases are selected mechanically from the classifier-vs-HMM table.

| Case | Trip | Classifier Top-1 | HMM Top-1 | Classifier continuity | HMM continuity | Classifier LCS-F1 | HMM LCS-F1 | Map |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| largest_top1_regression | 00000077 | 0.7951 | 0.6473 | 0.6696 | 0.9047 | 0.4401 | 0.2445 | `C:\Users\nandu\Desktop\ML-HACKATHON-Group-8\results\hmm\failure_maps\00000077_largest_top1_regression.html` |
| largest_continuity_recovery | 00000091 | 0.8913 | 0.8284 | 0.6605 | 0.8265 | 0.7103 | 0.6831 | `C:\Users\nandu\Desktop\ML-HACKATHON-Group-8\results\hmm\failure_maps\00000091_largest_continuity_recovery.html` |
| largest_sequence_regression | 00000040 | 0.9569 | 0.9104 | 0.7827 | 0.8904 | 0.7779 | 0.6184 | `C:\Users\nandu\Desktop\ML-HACKATHON-Group-8\results\hmm\failure_maps\00000040_largest_sequence_regression.html` |

## Interpretation checklist

- A Top-1 regression means smoothing selected a connected but wrong road; inspect junctions and parallel roads.
- A continuity recovery means the HMM removed an independent-classifier teleport.
- A sequence regression means continuity alone was not enough; inspect gaps, missing candidates, or over-smoothing.
