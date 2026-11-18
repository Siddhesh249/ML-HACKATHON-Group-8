import numpy as np

from pathsnap import metrics
from pathsnap.contracts import CandidateSet


def test_candidate_metrics_are_finite():
    y = np.array([1, 0, 0, 1])
    p = np.array([0.9, 0.1, 0.2, 0.8])
    out = metrics.candidate_metrics(y, p)
    assert out["n"] == 4
    assert out["precision"] == 1.0
    assert out["recall"] == 1.0
    assert 0.0 <= out["log_loss"] < 1.0
    assert out["macro_f1"] == 1.0


def test_top1_and_miss_rate():
    candidates = CandidateSet(
        offsets=np.array([0, 3, 5], dtype=np.int64),
        arc_ids=np.array([10, 11, 12, 20, 21], dtype=np.int64),
        distance_m=np.array([1.0, 2.0, 3.0, 1.0, 2.0]),
        projection_ratio=np.zeros(5),
        projection_xy=np.zeros((5, 2)),
        arc_heading_deg=np.zeros(5),
    )
    probs = np.array([0.8, 0.1, 0.1, 0.6, 0.4])
    labels = np.array([10, 99])
    confidence = np.array([True, True])
    top1, n_eval, correct = metrics.top1_accuracy(candidates, probs, labels, confidence)
    assert n_eval == 2
    assert correct == 1
    assert top1 == 0.5
    miss, n = metrics.candidate_miss_rate(candidates, labels, confidence)
    assert n == 2
    assert miss == 0.5


def test_lcs_and_sequence_metrics():
    assert metrics.lcs_length([1, 2, 3], [1, 3]) == 2
    out = metrics.sequence_prf(np.array([1, 2, 3, 4]), np.array([1, 3, 4]))
    assert out["lcs_match"] == 3
    assert 0 < out["lcs_f1"] <= 1


def test_dedupe_and_arc_set():
    assert metrics.dedupe_sequence([1, 1, 2, -1, 2, 3]).tolist() == [1, 2, 3]
    out = metrics.arc_set_prf(np.array([1, 2, 3]), np.array([2, 3, 4]))
    assert out["arc_precision"] == 2 / 3
    assert out["arc_recall"] == 2 / 3
