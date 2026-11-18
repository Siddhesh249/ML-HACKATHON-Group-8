"""Shared offline evaluation helpers for C's decoder experiments."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathsnap import candidates, config, features, hmm, io, metrics, models, scoring  # noqa: E402
from pathsnap.contracts import Alignment, CandidateSet, TripData  # noqa: E402


@dataclass
class PreparedTrip:
    trip: TripData
    scored_trip: TripData
    alignment: Alignment
    candidates: CandidateSet
    probabilities: np.ndarray
    model_version: str


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def prepare_trip(data_root: str | Path, artifact_dir: str | Path, trip_id: str) -> PreparedTrip:
    data_root = Path(data_root)
    artifact_dir = Path(artifact_dir)
    trip_dir = io.resolve_trip_dir(trip_id, data_root)
    trip = io.load_trip(trip_dir, with_route=False)
    payload = models.load_artifact(artifact_dir / "model.joblib")
    runtime = dict(payload.get("config", {}))
    generator = candidates.CandidateGenerator(
        trip.network,
        radius_m=float(runtime.get("candidate_radius_m", config.CANDIDATE_RADIUS_M)),
        top_k=int(runtime.get("candidate_top_k", config.CANDIDATE_TOP_K)),
    )
    candidate_set = generator.generate(trip.track)
    feature_matrix = features.build_features(
        trip,
        candidate_set,
        alignment=None,
        options=features.FeatureOptions(
            use_graph=bool(runtime.get("use_graph_features", config.USE_GRAPH_FEATURES)),
            smooth_window=int(runtime.get("smooth_window", config.SMOOTH_WINDOW)),
        ),
    )
    probabilities = np.asarray(payload["model"].predict_proba(feature_matrix.X)[:, 1], dtype=float)
    scored_trip, alignment = scoring.load_alignment(trip_dir)
    return PreparedTrip(
        trip=trip,
        scored_trip=scored_trip,
        alignment=alignment,
        candidates=candidate_set,
        probabilities=probabilities,
        model_version=str(payload.get("model_version", "unknown")),
    )


def _candidate_labels(candidate_set: CandidateSet, alignment: Alignment) -> np.ndarray:
    y = np.full(candidate_set.arc_ids.size, -1, dtype=np.int8)
    for obs in range(len(candidate_set)):
        if not alignment.confidence[obs] or alignment.arc_ids[obs] < 0:
            continue
        lo, hi = int(candidate_set.offsets[obs]), int(candidate_set.offsets[obs + 1])
        y[lo:hi] = (candidate_set.arc_ids[lo:hi] == alignment.arc_ids[obs]).astype(np.int8)
    return y


def evaluate_prepared(prepared: PreparedTrip, decoder_config: hmm.HMMConfig, mode: str) -> dict:
    start = time.perf_counter()
    prediction = hmm.decode(
        prepared.trip, prepared.candidates, prepared.probabilities, mode=mode, cfg=decoder_config
    )
    decode_ms = (time.perf_counter() - start) * 1000.0
    evaluation = scoring.evaluate_predictions(
        prepared.scored_trip, prepared.candidates, prepared.alignment, prediction.arc_ids
    )
    candidate = metrics.candidate_metrics(
        _candidate_labels(prepared.candidates, prepared.alignment), prepared.probabilities
    )
    return {
        "trip_id": prepared.trip.meta.trip_id,
        "mode": mode,
        "model_version": prepared.model_version,
        "n_observations": evaluation.n_observations,
        "decode_ms": decode_ms,
        "candidate_macro_f1": candidate.get("macro_f1", float("nan")),
        "candidate_log_loss": candidate.get("log_loss", float("nan")),
        **evaluation.metrics,
    }


def load_hmm_config(path: str | Path) -> hmm.HMMConfig:
    import json

    path = Path(path)
    if not path.exists():
        return hmm.HMMConfig()
    return hmm.HMMConfig(**json.loads(path.read_text(encoding="utf-8")))
