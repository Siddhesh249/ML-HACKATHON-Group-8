"""FastAPI application for Laptop 1.

Run:
    uvicorn backend.app:app --host 0.0.0.0 --port 8000 --log-level info
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from pathsnap import __version__, models
from pathsnap.contracts import DataError, ModelNotReady

from .schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TripsResponse,
    VersionResponse,
)
from .service import Predictor

logging.basicConfig(
    level=os.environ.get("PATHSNAP_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("pathsnap.api")


def _api_key_guard(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected = os.environ.get("PATHSNAP_API_KEY", "").strip()
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    return None


def create_app(predictor: Predictor | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if predictor is not None:
            app.state.predictor = predictor
        else:
            p = Predictor()
            p.load()
            app.state.predictor = p
        logger.info(
            "PathSnap API starting version=%s ready=%s model=%s",
            __version__,
            app.state.predictor.ready,
            app.state.predictor.model_version,
        )
        yield
        logger.info("PathSnap API shutting down")

    app = FastAPI(
        title="PathSnap API",
        version=__version__,
        description="Distributed map-matching and route reconstruction — Group 8",
        lifespan=lifespan,
    )

    origins = [o.strip() for o in os.environ.get("PATHSNAP_CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_logger(request: Request, call_next):
        request_id = uuid.uuid4().hex[:8]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "HTTP %s %s status=%d ms=%.1f rid=%s client=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
            request.client.host if request.client else "-",
        )
        return response

    def get_predictor(request: Request) -> Predictor:
        return request.app.state.predictor

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        state = request.app.state.predictor
        status = state.status()
        return HealthResponse(
            status="ok" if status["ready"] else "degraded",
            ready=bool(status["ready"]),
            model_version=status["model_version"],
            model_name=status["model_name"],
            feature_version=status["feature_version"],
            n_trips=int(status["n_trips"]),
            partition=status["partition"],
            message=status["message"],
        )

    @app.get("/version", response_model=VersionResponse)
    def version(request: Request) -> VersionResponse:
        state = request.app.state.predictor
        return VersionResponse(
            api_version=__version__,
            git_commit=models.git_commit(),
            python=sys.version.split()[0] + f" ({platform.system()})",
            model_version=state.model_version,
        )

    @app.get("/trips", response_model=TripsResponse, dependencies=[Depends(_api_key_guard)])
    def trips(request: Request) -> TripsResponse:
        state = request.app.state.predictor
        partition, rows = state.list_trips()
        return TripsResponse(partition=partition, count=len(rows), trips=rows)

    @app.post(
        "/predict",
        response_model=PredictResponse,
        dependencies=[Depends(_api_key_guard)],
    )
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        state = request.app.state.predictor
        try:
            result = state.predict(payload.trip_id, mode=payload.mode)
        except ModelNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except DataError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - return a clear server error
            logger.exception("Prediction failed trip=%s", payload.trip_id)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
        return PredictResponse(**result)

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PATHSNAP_API_PORT", "8000")),
        log_level="info",
    )
