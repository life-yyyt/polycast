"""Pipeline orchestration.

Order matters and is intentional:

    retrieve -> independent analysts (parallel) -> red team -> aggregate -> persist

The red team runs *after* aggregation inputs exist but *before* the number is
fixed, and analysts never see each other's answers. Running the panel
sequentially with shared context collapses disagreement and produces the
comfortable, overconfident forecasts this design is trying to avoid.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from .agents import (
    DEFAULT_PANEL,
    RED_TEAM,
    aggregate,
    build_analyst_prompt,
    build_red_team_prompt,
)
from .audit import AuditTrail, ForecastStore
from .providers import JsonProvider
from .retrieval import TfidfRetriever
from .schemas import (
    AnalystView,
    Critique,
    Forecast,
    Question,
    new_id,
)


class ForecastPipeline:
    def __init__(
        self,
        provider: JsonProvider,
        retriever: TfidfRetriever,
        store: ForecastStore | None = None,
        audit_dir: str | Path = "runs",
        top_k: int = 5,
    ) -> None:
        self.provider = provider
        self.retriever = retriever
        self.store = store
        self.audit_dir = audit_dir
        self.top_k = top_k

    async def run(self, question: Question) -> Forecast:
        trace_id = new_id()
        audit = AuditTrail(trace_id, self.audit_dir)
        audit.record("question.received", question.model_dump(mode="json"))

        evidence = self.retriever.search(question.text, k=self.top_k)
        audit.record(
            "retrieval.completed",
            {
                "k": len(evidence),
                "doc_ids": [e.doc_id for e in evidence],
                "scores": {e.doc_id: round(e.score, 4) for e in evidence},
            },
        )

        raw_views = await asyncio.gather(
            *(
                self.provider.complete_json(
                    p.system_prompt, build_analyst_prompt(question, evidence, p.name)
                )
                for p in DEFAULT_PANEL
            )
        )
        views = [AnalystView.model_validate(v) for v in raw_views]
        for view in views:
            audit.record("analyst.completed", view.model_dump(mode="json"))

        panel_mean = sum(v.probability for v in views) / len(views)
        critique = Critique.model_validate(
            await self.provider.complete_json(
                RED_TEAM.system_prompt, build_red_team_prompt(question, views, panel_mean)
            )
        )
        audit.record("red_team.completed", critique.model_dump(mode="json"))

        probability, lower, upper, disagreement = aggregate(views, critique)
        forecast = Forecast(
            forecast_id=trace_id,
            question=question,
            probability=probability,
            lower=lower,
            upper=upper,
            disagreement=disagreement,
            views=views,
            evidence=evidence,
            critique=critique,
        )
        audit.record(
            "forecast.aggregated",
            {
                "panel_mean": round(panel_mean, 4),
                "shift_applied": critique.probability_shift,
                "probability": probability,
                "interval": [lower, upper],
                "disagreement": disagreement,
            },
        )

        if self.store:
            self.store.save(forecast)
            audit.record("forecast.persisted", {"forecast_id": forecast.forecast_id})
        return forecast
