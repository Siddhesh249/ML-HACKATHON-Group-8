"""Ground-truth isolation tests.

The prediction path must never attach the reference route. Scoring is the first
place allowed to read it.
"""

from __future__ import annotations

from backend.service import Predictor
from pathsnap import scoring


def test_prediction_loader_does_not_attach_route(synthetic_root):
    root, trip_id, _ = synthetic_root
    predictor = Predictor(data_root=root)
    trip = predictor._load_for_prediction(trip_id)
    assert trip.route is None
    assert (root / "student_test" / trip_id / f"{trip_id}.route").exists()


def test_scoring_is_the_only_route_reader(synthetic_root):
    root, trip_id, _ = synthetic_root
    trip, alignment = scoring.load_alignment(root / "student_test" / trip_id)
    assert trip.route is not None
    assert alignment.confidence.size == len(trip.track)


def test_predictor_refuses_missing_artifact(synthetic_root):
    root, trip_id, _ = synthetic_root
    predictor = Predictor(data_root=root, artifact_dir=root / "no-artifacts").load()
    assert predictor.ready is False
    try:
        predictor.predict(trip_id)
    except Exception as exc:  # ModelNotReady
        assert "artifact" in str(exc).lower() or "not ready" in str(exc).lower()
        return
    raise AssertionError("prediction should fail without an artifact")
