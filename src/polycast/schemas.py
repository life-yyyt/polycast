"""Typed contracts shared by the pipeline, the API and the audit log.

Everything that crosses a process boundary is a pydantic model: if an LLM
returns something that does not validate, the run fails loudly instead of
silently producing a miscalibrated number downstream.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Horizon(StrEnum):
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"


class Question(BaseModel):
    """A resolvable binary question.

    `resolution_criteria` is mandatory on purpose. Questions without an explicit
    resolution rule are the single largest source of garbage forecasts: analysts
    end up answering slightly different questions and the spread looks like
    genuine uncertainty.
    """

    text: str
    resolution_criteria: str
    resolve_by: datetime
    horizon: Horizon = Horizon.MONTHS
    tags: list[str] = Field(default_factory=list)
    reference_class: str | None = None

    @field_validator("text")
    @classmethod
    def _text_not_empty(cls, v: str) -> str:
        if len(v.strip()) < 12:
            raise ValueError("question text is too short to be resolvable")
        return v.strip()


class Document(BaseModel):
    """A retrievable piece of evidence."""

    doc_id: str
    text: str
    source: str
    published_at: datetime
    url: str | None = None


class Evidence(Document):
    """A document after retrieval scoring."""

    score: float = 0.0
    freshness: float = 1.0


class AnalystView(BaseModel):
    persona: str
    probability: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class Critique(BaseModel):
    """Red-team pass over the analyst panel."""

    overlooked: list[str] = Field(default_factory=list)
    probability_shift: float = Field(default=0.0, ge=-0.3, le=0.3)
    notes: str = ""


class Forecast(BaseModel):
    forecast_id: str = Field(default_factory=new_id)
    question: Question
    probability: float = Field(ge=0.0, le=1.0)
    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    disagreement: float = 0.0
    views: list[AnalystView] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    critique: Critique | None = None
    created_at: datetime = Field(default_factory=utcnow)


class ResolvedQuestion(BaseModel):
    """A question whose outcome is known, used for scoring."""

    question: Question
    outcome: int = Field(ge=0, le=1)


class CalibrationBin(BaseModel):
    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_frequency: float


class CalibrationReport(BaseModel):
    n: int
    brier: float
    log_loss: float
    base_rate_brier: float
    bins: list[CalibrationBin]

    @property
    def skill_vs_base_rate(self) -> float:
        """1 - brier/base_rate_brier. Positive means we beat the base rate."""
        if self.base_rate_brier == 0:
            return 0.0
        return 1.0 - self.brier / self.base_rate_brier
