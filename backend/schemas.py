"""FastAPI schemas (pydantic v2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    ready: bool
    model_version: str | None = None
    model_name: str | None = None
    feature_version: str | None = None
    n_trips: int = 0
    partition: str | None = None
    message: str | None = None


class VersionResponse(BaseModel):
    api_version: str
    git_commit: str
    python: str
    model_version: str | None = None


class TripSummary(BaseModel):
    trip_id: str
    partition: str
    n_points: int
    n_nodes: int
    n_arcs: int
    n_route_arcs: int
    has_route: bool
    bbox: list[float]


class TripsResponse(BaseModel):
    partition: str
    count: int
    trips: list[TripSummary]


class PredictRequest(BaseModel):
    trip_id: str = Field(pattern=r"^[0-9]{8}$", description="Eight-digit trip ID")
    mode: Literal["hmm", "classifier"] = "hmm"


class PredictResponse(BaseModel):
    trip_id: str
    model_version: str
    model_name: str
    mode: str
    metrics: dict[str, float | None]
    timings_ms: dict[str, float]
    n_observations: int
    n_matched: int
    predicted_route: list[list[float]]
    observed_points: list[list[float]]
    route_arc_count: int
