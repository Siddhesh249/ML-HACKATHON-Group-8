"""Evaluation metrics used by training, decoding and the API.

Definitions are part of the project contract (see docs/EXPERIMENTS.md).
"""

from __future__ import annotations

import difflib

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from .contracts import CandidateSet, Network


# ---------------------------------------------------------------------------
# Candidate-level metrics
# ---------------------------------------------------------------------------
def candidate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Binary metrics over candidate pairs (positive = reference arc)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = y_true >= 0
    y = y_true[mask]
    p = np.clip(y_prob[mask], 1e-7, 1 - 1e-7)
    if y.size == 0:
        return {"n": 0.0}
    y_hat = (p >= 0.5).astype(int)
    out = {
        "n": float(y.size),
        "accuracy": float(accuracy_score(y, y_hat)),
        "precision": float(precision_score(y, y_hat, zero_division=0)),
        "recall": float(recall_score(y, y_hat, zero_division=0)),
        "f1": float(f1_score(y, y_hat, zero_division=0)),
        "macro_f1": float(f1_score(y, y_hat, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }
    return out


# ---------------------------------------------------------------------------
# Observation-level metrics
# ---------------------------------------------------------------------------
def top1_accuracy(
    candidates: CandidateSet,
    probs: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
) -> tuple[float, int, int]:
    """Argmax candidate per observation vs the aligned reference arc.

    A labelled observation whose reference arc is missing from the candidate
    set counts as incorrect. This is the honest ceiling; report the miss rate
    next to it so the two numbers are not confused.
    """
    n_eval = 0
    n_correct = 0
    for i in range(len(candidates)):
        if not confidence[i] or labels[i] < 0:
            continue
        lo, hi = int(candidates.offsets[i]), int(candidates.offsets[i + 1])
        if hi <= lo:
            n_eval += 1
            continue
        best = int(candidates.arc_ids[lo + int(np.argmax(probs[lo:hi]))])
        n_eval += 1
        if best == int(labels[i]):
            n_correct += 1
    return (n_correct / n_eval if n_eval else float("nan")), n_eval, n_correct


def pointwise_accuracy(
    predicted_arcs: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
) -> tuple[float, int, int]:
    n_eval = 0
    n_correct = 0
    for pred, truth, conf in zip(predicted_arcs, labels, confidence):
        if not conf or truth < 0:
            continue
        n_eval += 1
        if int(pred) == int(truth):
            n_correct += 1
    return (n_correct / n_eval if n_eval else float("nan")), n_eval, n_correct


def candidate_miss_rate(
    candidates: CandidateSet,
    labels: np.ndarray,
    confidence: np.ndarray,
) -> tuple[float, int]:
    n_eval = 0
    n_miss = 0
    for i in range(len(candidates)):
        if not confidence[i] or labels[i] < 0:
            continue
        n_eval += 1
        lo, hi = int(candidates.offsets[i]), int(candidates.offsets[i + 1])
        if labels[i] not in candidates.arc_ids[lo:hi]:
            n_miss += 1
    return (n_miss / n_eval if n_eval else float("nan")), n_eval


# ---------------------------------------------------------------------------
# Route / sequence metrics
# ---------------------------------------------------------------------------
def dedupe_sequence(arc_ids: np.ndarray) -> np.ndarray:
    out: list[int] = []
    for arc in np.asarray(arc_ids).ravel():
        arc = int(arc)
        if arc < 0:
            continue
        if not out or out[-1] != arc:
            out.append(arc)
    return np.asarray(out, dtype=np.int64)


def lcs_length(a: np.ndarray, b: np.ndarray) -> int:
    """Length of the longest common subsequence (via matching blocks)."""
    a = [int(x) for x in np.asarray(a).ravel()]
    b = [int(x) for x in np.asarray(b).ravel()]
    if not a or not b:
        return 0
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return int(sum(block.size for block in matcher.get_matching_blocks()))


def sequence_prf(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    pred = dedupe_sequence(predicted)
    ref = dedupe_sequence(reference)
    match = lcs_length(pred, ref)
    precision = match / len(pred) if len(pred) else 0.0
    recall = match / len(ref) if len(ref) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "lcs_match": float(match),
        "lcs_precision": float(precision),
        "lcs_recall": float(recall),
        "lcs_f1": float(f1),
        "exact_match": float(len(pred) == len(ref) and match == len(ref)),
        "n_pred_arcs": float(len(pred)),
        "n_ref_arcs": float(len(ref)),
    }


def arc_set_prf(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    pred = set(int(x) for x in np.asarray(predicted).ravel() if int(x) >= 0)
    ref = set(int(x) for x in np.asarray(reference).ravel() if int(x) >= 0)
    tp = len(pred & ref)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "arc_precision": float(precision),
        "arc_recall": float(recall),
        "arc_f1": float(f1),
    }


def route_continuity(
    arc_ids: np.ndarray,
    network: Network,
    node_xy: np.ndarray | None = None,
    slack_m: float = 1.0,
) -> dict[str, float]:
    """Fraction of consecutive predicted arcs that are graph-connected."""
    seq = dedupe_sequence(arc_ids)
    if len(seq) < 2:
        return {"strict_continuity": float("nan"), "relaxed_continuity": float("nan"), "n_transitions": 0.0}

    arcs = network.arcs
    strict = 0
    relaxed = 0
    for a, b in zip(seq[:-1], seq[1:]):
        end = int(arcs[a, 1])
        start = int(arcs[b, 0])
        if end == start:
            strict += 1
            relaxed += 1
        elif node_xy is not None:
            if np.linalg.norm(node_xy[start] - node_xy[end]) <= slack_m:
                relaxed += 1
    n = len(seq) - 1
    return {
        "strict_continuity": float(strict / n),
        "relaxed_continuity": float(relaxed / n),
        "n_transitions": float(n),
    }


def sequence_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    out = sequence_prf(predicted, reference)
    out.update(arc_set_prf(predicted, reference))
    return out
