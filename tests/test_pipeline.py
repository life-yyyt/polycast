import json
from datetime import datetime, timezone

import pytest

from polycast.agents import aggregate
from polycast.corpus import build_retriever
from polycast.pipeline import ForecastPipeline
from polycast.providers import MockProvider
from polycast.schemas import AnalystView, Question


def make_question() -> Question:
    return Question(
        text="Will the central bank cut the policy rate at the next scheduled meeting?",
        resolution_criteria="Resolved yes if the published decision confirms a reduction.",
        resolve_by=datetime(2026, 12, 18, tzinfo=timezone.utc),
        reference_class="scheduled policy meetings",
    )


@pytest.fixture
def pipeline(tmp_path):
    return ForecastPipeline(
        provider=MockProvider(),
        retriever=build_retriever(),
        audit_dir=tmp_path / "runs",
    )


async def test_pipeline_produces_a_bounded_probability(pipeline):
    forecast = await pipeline.run(make_question())
    assert 0.0 < forecast.probability < 1.0
    assert forecast.lower <= forecast.probability <= forecast.upper
    assert len(forecast.views) == 3
    assert {v.persona for v in forecast.views} == {"base_rate", "bull", "bear"}


async def test_pipeline_retrieves_evidence(pipeline):
    forecast = await pipeline.run(make_question())
    assert forecast.evidence, "expected at least one retrieved document"
    assert all(0.0 <= e.score <= 1.0 for e in forecast.evidence)


async def test_pipeline_writes_an_audit_trace(pipeline, tmp_path):
    forecast = await pipeline.run(make_question())
    trace = tmp_path / "runs" / f"{forecast.forecast_id}.jsonl"
    assert trace.exists()
    steps = [json.loads(line)["step"] for line in trace.read_text(encoding="utf-8").splitlines()]
    assert "question.received" in steps
    assert "retrieval.completed" in steps
    assert "analyst.completed" in steps
    assert "red_team.completed" in steps
    assert "forecast.aggregated" in steps


async def test_pipeline_persists_forecast(tmp_path):
    from polycast.audit import ForecastStore

    store = ForecastStore(tmp_path / "polycast.db")
    pipeline = ForecastPipeline(
        provider=MockProvider(),
        retriever=build_retriever(),
        store=store,
        audit_dir=tmp_path / "runs",
    )
    forecast = await pipeline.run(make_question())
    assert store.count() == 1
    assert store.get(forecast.forecast_id) is not None


def test_aggregate_of_unanimous_panel_has_no_disagreement():
    views = [
        AnalystView(persona="a", probability=0.6, reasoning="x"),
        AnalystView(persona="b", probability=0.6, reasoning="y"),
        AnalystView(persona="c", probability=0.6, reasoning="z"),
    ]
    prob, lower, upper, disagreement = aggregate(views)
    assert prob == 0.6
    assert disagreement == 0.0
    assert lower < prob < upper


def test_aggregate_trims_outliers_when_panel_is_large():
    views = [
        AnalystView(persona="a", probability=0.10, reasoning="x"),
        AnalystView(persona="b", probability=0.50, reasoning="y"),
        AnalystView(persona="c", probability=0.55, reasoning="z"),
        AnalystView(persona="d", probability=0.95, reasoning="w"),
    ]
    prob, _, _, disagreement = aggregate(views)
    assert 0.4 < prob < 0.7
    assert disagreement > 0.3


def test_aggregate_applies_red_team_shift():
    from polycast.schemas import Critique

    views = [
        AnalystView(persona="a", probability=0.5, reasoning="x"),
        AnalystView(persona="b", probability=0.5, reasoning="y"),
    ]
    critique = Critique(overlooked=["correlated sources"], probability_shift=-0.1)
    prob, _, _, _ = aggregate(views, critique)
    assert prob == 0.4


def test_aggregate_never_returns_absolute_zero_or_one():
    views = [AnalystView(persona="a", probability=0.0, reasoning="x")]
    prob, lower, upper, _ = aggregate(views)
    assert 0.0 < prob < 1.0
    assert lower >= 0.0 and upper <= 1.0
