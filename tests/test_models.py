"""Tests for B's training helpers: Top-1, feature groups, grouped CV."""

from __future__ import annotations

import numpy as np

from pathsnap import config, dataset, features, models


def _toy(seed: int = 0, n_trips: int = 6, obs_per_trip: int = 20):
    rng = np.random.default_rng(seed)
    row_obs, row_arc, y = [], [], []
    labels, confidence, obs_trip = [], [], []
    obs = 0
    for t in range(n_trips):
        for _ in range(obs_per_trip):
            k = int(rng.integers(0, 5))  # zero-candidate observations included
            arcs = rng.choice(100, size=k, replace=False) if k else np.empty(0, int)
            truth = int(arcs[rng.integers(0, k)]) if k and rng.random() < 0.9 else int(rng.integers(100, 200))
            labels.append(truth)
            confidence.append(bool(rng.random() < 0.85))
            obs_trip.append(t)
            for a in arcs:
                row_obs.append(obs)
                row_arc.append(int(a))
                y.append(int(a == truth))
            obs += 1
    row_obs = np.asarray(row_obs, dtype=np.int64)
    counts = np.bincount(row_obs, minlength=obs)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    obs_trip = np.asarray(obs_trip)
    return {
        "row_obs": row_obs,
        "row_arc": np.asarray(row_arc),
        "y": np.asarray(y),
        "obs_offsets": offsets,
        "labels": np.asarray(labels),
        "confidence": np.asarray(confidence),
        "obs_trip": obs_trip,
        "row_trip": obs_trip[row_obs],
        "probs": rng.random(row_obs.size),
    }


def test_top1_by_mask_matches_reference_loop():
    d = _toy()
    ref = dataset.top1_from_probs(d["probs"], d["obs_offsets"], d["row_arc"], d["labels"], d["confidence"])
    vec = dataset.top1_by_mask(d["probs"], d["row_obs"], d["row_arc"], d["labels"], d["confidence"])
    assert ref == vec


def test_top1_by_mask_subset_ignores_nan_outside_mask():
    d = _toy(seed=1)
    probs = d["probs"].copy()
    keep_trips = np.array([0, 2])
    probs[~np.isin(d["row_trip"], keep_trips)] = np.nan
    mask = np.isin(d["obs_trip"], keep_trips)
    conf = d["confidence"] & mask
    ref = dataset.top1_from_probs(np.nan_to_num(probs, nan=-1.0), d["obs_offsets"], d["row_arc"], d["labels"], conf)
    vec = dataset.top1_by_mask(probs, d["row_obs"], d["row_arc"], d["labels"], d["confidence"], mask)
    assert ref == vec


def test_feature_groups_partition_the_contract():
    names = [f for group in features.FEATURE_GROUPS.values() for f in group]
    assert sorted(names) == sorted(config.FEATURE_NAMES)
    assert len(names) == len(set(names))
    assert features.feature_indices(["candidate_count", "log_distance"]) == [0, 17]


def test_cv_split_table_keeps_trips_in_one_fold():
    d = _toy(n_trips=10)
    trip_ids = np.asarray([f"{i:08d}" for i in range(10)])
    table = dataset.cv_split_table(trip_ids, d["row_trip"], n_splits=5)
    assert len(table) == 10
    assert len({r["trip_id"] for r in table}) == 10
    assert {r["fold"] for r in table} == set(range(5))


def test_cross_validate_oof_grouped_and_scored():
    rng = np.random.default_rng(3)
    n = 600
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0.8).astype(np.int8)
    y[:20] = -1
    groups = np.repeat(np.arange(6), n // 6)
    seen = []

    def scorer(fold, est, train_idx, test_idx, prob):
        assert not set(groups[train_idx]) & set(groups[test_idx])
        seen.append(fold)
        return {"custom": float(prob.mean())}

    oof, folds, _ = models.cross_validate_oof(
        X, y, groups, "logistic_regression", n_splits=3, fold_scorer=scorer
    )
    assert np.isfinite(oof).all()
    assert seen == [0, 1, 2]
    assert all("custom" in f for f in folds)
