"""Prediction service: artifact loading, live inference and server-side scoring.

Ground-truth safety contract
----------------------------
``predict`` calls :meth:`Predictor._load_for_prediction`, which loads a trip
with ``with_route=False``. The reference ``.route`` is opened only after the
predicted arc sequence exists, inside ``pathsnap.scoring``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from pathsnap import __version__, candidates as cand_mod, config, features, hmm, io, models, scoring
from pathsnap.contracts import DataError, ModelNotReady, TripData

logger = logging.getLogger("pathsnap.predictor")


class Predictor:
    def __init__(
        self,
        data_root: str | Path | None = None,
        artifact_dir: str | Path | None = None,
        evaluation_partition: str | None = None,
        fallback_partition: str | None = None,
    ) -> None:
        self.data_root = Path(data_root) if data_root is not None else config.DATA_ROOT
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else config.ARTIFACT_DIR
        self.evaluation_partition = evaluation_partition or config.EVALUATION_PARTITION
        self.fallback_partition = fallback_partition or config.FALLBACK_PARTITION
        self.artifact: dict | None = None
        self.hmm_config = hmm.HMMConfig()
        self.ready = False
        self.load_error: str | None = None
        self.partition: str | None = None
        self._trips_loaded = False
        self._n_trips = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self) -> "Predictor":
        artifact_path = self.artifact_dir / "model.joblib"
        if not artifact_path.exists():
            self.ready = False
            self.load_error = f"Missing artifact: {artifact_path}. Run `make export`."
            logger.warning(self.load_error)
            return self
        try:
            payload = models.load_artifact(artifact_path)
            feature_names = tuple(payload.get("feature_names", ()))
            if feature_names and feature_names != tuple(config.FEATURE_NAMES):
                raise ValueError(
                    "Artifact feature contract does not match config.FEATURE_NAMES. "
                    "Re-export the model after updating the contract."
                )
            self.artifact = payload
            hmm_path = self.artifact_dir / "hmm_config.json"
            if hmm_path.exists():
                with open(hmm_path, encoding="utf-8") as fh:
                    self.hmm_config = hmm.HMMConfig(**json.load(fh))
            self.ready = True
            self.load_error = None
            logger.info(
                "Loaded artifact model=%s version=%s features=%d",
                payload.get("model_name"),
                payload.get("model_version"),
                len(feature_names),
            )
        except Exception as exc:  # noqa: BLE001 - health must report the reason
            self.ready = False
            self.load_error = f"Could not load artifact: {exc}"
            logger.exception("Artifact load failed")
        return self

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    @property
    def model_version(self) -> str | None:
        return None if self.artifact is None else str(self.artifact.get("model_version", "unknown"))

    @property
    def model_name(self) -> str | None:
        return None if self.artifact is None else str(self.artifact.get("model_name", "unknown"))

    @property
    def n_trips(self) -> int:
        if not self._trips_loaded:
            self.list_trips()
        return self._n_trips

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "model_version": self.model_version,
            "model_name": self.model_name,
            "feature_version": config.FEATURE_VERSION,
            "n_trips": self.n_trips,
            "partition": self.partition,
            "message": self.load_error,
        }

    def list_trips(self) -> tuple[str, list[dict]]:
        metas = io.discover_trips(self.data_root, partition=self.evaluation_partition)
        used = self.evaluation_partition
        if not metas:
            metas = io.discover_trips(self.data_root, partition=self.fallback_partition)
            used = self.fallback_partition
        self.partition = used
        self._trips_loaded = True
        self._n_trips = len(metas)
        return used, [m.to_dict() for m in metas]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _load_for_prediction(self, trip_id: str) -> TripData:
        """Load only the inputs required for prediction (never ``.route``)."""
        directory = io.resolve_trip_dir(trip_id, self.data_root)
        trip = io.load_trip(directory, with_route=False)
        if trip.route is not None:  # defensive: load_trip must not attach truth
            raise RuntimeError("Ground-truth route leaked into the prediction path")
        return trip

    def _use_graph_features(self) -> bool:
        cfg = {} if self.artifact is None else self.artifact.get("config", {})
        return bool(cfg.get("use_graph_features", config.USE_GRAPH_FEATURES))

    def _candidate_radius(self) -> float:
        cfg = {} if self.artifact is None else self.artifact.get("config", {})
        return float(cfg.get("candidate_radius_m", config.CANDIDATE_RADIUS_M))

    def _top_k(self) -> int:
        cfg = {} if self.artifact is None else self.artifact.get("config", {})
        return int(cfg.get("candidate_top_k", config.CANDIDATE_TOP_K))

    def predict(self, trip_id: str, mode: str = "hmm") -> dict:
        if not self.ready or self.artifact is None:
            raise ModelNotReady(self.load_error or "Model is not ready")
        if mode not in {"hmm", "classifier"}:
            raise ValueError("mode must be 'hmm' or 'classifier'")

        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        t0 = time.perf_counter()
        trip = self._load_for_prediction(trip_id)
        timings["load_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        generator = cand_mod.CandidateGenerator(
            trip.network, radius_m=self._candidate_radius(), top_k=self._top_k()
        )
        candidates = generator.generate(trip.track)
        timings["candidates_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        feature_matrix = features.build_features(
            trip,
            candidates,
            alignment=None,
            options=features.FeatureOptions(use_graph=self._use_graph_features()),
        )
        timings["features_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        estimator = self.artifact["model"]
        probs = np.asarray(estimator.predict_proba(feature_matrix.X)[:, 1], dtype=float)
        timings["model_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        prediction = hmm.decode(trip, candidates, probs, mode=mode, cfg=self.hmm_config)
        prediction.model_version = self.model_version or "unknown"
        prediction.timings_ms = timings
        timings["decode_ms"] = (time.perf_counter() - t0) * 1000

        # ---- scoring: the first point at which the reference route is read ----
        t0 = time.perf_counter()
        directory = io.resolve_trip_dir(trip_id, self.data_root)
        trip_with_route, alignment = scoring.load_alignment(directory)
        evaluation = scoring.evaluate_predictions(
            trip_with_route, candidates, alignment, prediction.arc_ids
        )
        timings["scoring_ms"] = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = (time.perf_counter() - total_start) * 1000

        route_arcs = prediction.route_arcs()
        predicted_route = scoring.polyline_from_arcs(trip.network, route_arcs)
        observed = self._observed_points(trip)
        metrics = self._clean_metrics(evaluation.metrics)

        logger.info(
            "PREDICT trip=%s points=%d candidates=%d mode=%s top1=%s "
            "continuity=%s latency_ms=%.1f model=%s",
            trip_id,
            len(trip.track),
            len(candidates.arc_ids),
            mode,
            metrics.get("point_top1"),
            metrics.get("strict_continuity"),
            timings["total_ms"],
            self.model_version,
        )

        return {
            "trip_id": trip_id,
            "model_version": self.model_version or "unknown",
            "model_name": self.model_name or "unknown",
            "mode": mode,
            "metrics": metrics,
            "timings_ms": {k: float(v) for k, v in timings.items()},
            "n_observations": evaluation.n_observations,
            "n_matched": evaluation.n_matched,
            "predicted_route": predicted_route,
            "observed_points": observed,
            "route_arc_count": int(route_arcs.size),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _observed_points(trip: TripData, max_points: int = 3000) -> list[list[float]]:
        lon = trip.track.lonlat[:, 0]
        lat = trip.track.lonlat[:, 1]
        idx = np.linspace(0, len(lon) - 1, min(max_points, len(lon))).astype(int)
        return [[float(lon[i]), float(lat[i])] for i in idx]

    @staticmethod
    def _clean_metrics(metrics: dict) -> dict:
        cleaned: dict[str, float | None] = {}
        for key, value in metrics.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                cleaned[key] = None
                continue
            cleaned[key] = v if math.isfinite(v) else None
        return cleaned


__all__ = ["Predictor", "__version__"]
