"""API contract tests with an injected fake predictor (no dataset needed)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import create_app


class FakePredictor:
    ready = True
    model_version = "test-version"
    model_name = "fake-model"
    load_error = None

    def status(self):
        return {
            "ready": True,
            "model_version": self.model_version,
            "model_name": self.model_name,
            "feature_version": "v1",
            "n_trips": 10,
            "partition": "evaluator_test",
            "message": None,
        }

    def list_trips(self):
        return (
            "evaluator_test",
            [
                {
                    "trip_id": "00000000",
                    "partition": "evaluator_test",
                    "n_points": 100,
                    "n_nodes": 50,
                    "n_arcs": 100,
                    "n_route_arcs": 10,
                    "has_route": True,
                    "bbox": [8.0, 8.1, 49.0, 49.1],
                }
            ],
        )

    def predict(self, trip_id: str, mode: str = "hmm"):
        return {
            "trip_id": trip_id,
            "model_version": self.model_version,
            "model_name": self.model_name,
            "mode": mode,
            "metrics": {
                "point_top1": 0.75,
                "candidate_miss_rate": 0.01,
                "strict_continuity": 0.9,
                "lcs_f1": 0.8,
                "arc_f1": 0.7,
            },
            "timings_ms": {"total_ms": 12.3},
            "n_observations": 100,
            "n_matched": 100,
            "predicted_route": [[8.0, 49.0], [8.1, 49.1]],
            "observed_points": [[8.0, 49.0], [8.1, 49.1]],
            "route_arc_count": 10,
        }


def client() -> TestClient:
    return TestClient(create_app(FakePredictor()))


def test_health_and_version():
    with client() as c:
        health = c.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["ready"] is True
        assert body["n_trips"] == 10
        assert body["partition"] == "evaluator_test"

        version = c.get("/version")
        assert version.status_code == 200
        assert version.json()["model_version"] == "test-version"


def test_trips_have_no_ground_truth_geometry():
    with client() as c:
        response = c.get("/trips")
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 1
        trip = body["trips"][0]
        assert trip["trip_id"] == "00000000"
        assert "route" not in trip
        assert "reference_route" not in trip


def test_predict_contract_and_no_reference_route():
    with client() as c:
        response = c.post("/predict", json={"trip_id": "00000000", "mode": "hmm"})
        assert response.status_code == 200
        body = response.json()
        assert body["trip_id"] == "00000000"
        assert body["mode"] == "hmm"
        assert "metrics" in body
        assert "predicted_route" in body
        assert "observed_points" in body
        assert "reference_route" not in body
        assert "ground_truth" not in body


def test_predict_rejects_invalid_trip_id():
    with client() as c:
        response = c.post("/predict", json={"trip_id": "../etc/passwd"})
        assert response.status_code == 422
