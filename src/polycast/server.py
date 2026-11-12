"""HTTP API.

FastAPI over the same pipeline the CLI uses. Two things worth noting:

* The provider is chosen once at startup from environment variables, so a
  deployment either uses a hosted model or the offline stub and never mixes
  them per request.
* Errors are surfaced as 422 with the validation message attached. A forecast
  pipeline that silently degrades is worse than one that refuses to answer.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .audit import ForecastStore
from .corpus import build_retriever
from .pipeline import ForecastPipeline
from .providers import MockProvider, OpenAICompatibleProvider
from .schemas import Forecast, Horizon, Question
from .simulation import run_simulation


def build_pipeline() -> ForecastPipeline:
    provider: Any
    if os.getenv("POLYCAST_MODEL"):
        provider = OpenAICompatibleProvider(
            model=os.environ["POLYCAST_MODEL"],
            api_key=os.getenv("POLYCAST_API_KEY"),
            base_url=os.getenv("POLYCAST_BASE_URL", "https://api.openai.com/v1"),
        )
    else:
        provider = MockProvider()
    return ForecastPipeline(
        provider=provider,
        retriever=build_retriever(os.getenv("POLYCAST_CORPUS")),
        store=ForecastStore(os.getenv("POLYCAST_DB", "polycast.db")),
        audit_dir=os.getenv("POLYCAST_AUDIT_DIR", "runs"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = build_pipeline()
    yield


app = FastAPI(
    title="polycast",
    version="0.1.0",
    summary="Multi-agent forecasting with calibrated probabilities and an audit trail",
    lifespan=lifespan,
)


class ForecastRequest(BaseModel):
    text: str = Field(..., min_length=12)
    resolution_criteria: str = Field(..., min_length=8)
    resolve_by: datetime
    horizon: Horizon = Horizon.MONTHS
    reference_class: str | None = None
    tags: list[str] = Field(default_factory=list)


class SimulationRequest(BaseModel):
    n_agents: int = Field(40, ge=2, le=500)
    truth: float = Field(0.7, ge=0.01, le=0.99)
    rounds: int = Field(60, ge=1, le=500)
    social_weight: float = Field(0.3, ge=0.0, le=1.0)
    topology: str = "dense"
    signal_noise: float = Field(0.12, ge=0.001, le=1.0)
    seed: int = 7


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.post("/forecast", response_model=Forecast)
async def create_forecast(req: ForecastRequest) -> Forecast:
    question = Question(**req.model_dump())
    forecast = await app.state.pipeline.run(question)
    return forecast


@app.get("/forecast/{forecast_id}", response_model=Forecast)
async def get_forecast(forecast_id: str) -> Forecast:
    store = app.state.pipeline.store
    forecast = store.get(forecast_id) if store else None
    if forecast is None:
        raise HTTPException(status_code=404, detail="forecast not found")
    return forecast


@app.get("/forecasts")
async def list_forecasts(limit: int = 20, le: int = 100) -> dict[str, Any]:
    store = app.state.pipeline.store
    return {"items": store.recent(limit) if store else [], "total": store.count() if store else 0}


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    store = app.state.pipeline.store
    rows = store.recent(100) if store else []
    disagreements = [r["disagreement"] for r in rows] if rows else []
    return {
        "forecasts_stored": store.count() if store else 0,
        "mean_disagreement": round(sum(disagreements) / len(disagreements), 4)
        if disagreements
        else None,
        "provider": "hosted" if os.getenv("POLYCAST_MODEL") else "offline-stub",
    }


@app.post("/simulate")
async def simulate(req: SimulationRequest) -> dict[str, Any]:
    result = run_simulation(
        n_agents=req.n_agents,
        truth=req.truth,
        rounds=req.rounds,
        social_weight=req.social_weight,
        topology=req.topology,
        signal_noise=req.signal_noise,
        seed=req.seed,
    )
    return {
        "truth": result.truth,
        "final_mean": result.final_mean,
        "final_spread": result.final_spread,
        "converged_round": result.converged_round,
        "mean_series": result.mean_series,
        "spread_series": result.spread_series,
    }
