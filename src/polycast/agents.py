"""The analyst panel and the way we combine it.

Diversity here is by construction, not by prompt decoration: the base-rate
analyst is forbidden from reading narrative detail into the prior, the bull and
the bear are explicitly rewarded for finding disconfirming evidence for each
other, and the red team runs after the panel so it argues against a concrete
number rather than a strawman.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .schemas import AnalystView, Critique, Evidence, Question


@dataclass(frozen=True)
class Persona:
    name: str
    system_prompt: str
    weight: float = 1.0


DEFAULT_PANEL: tuple[Persona, ...] = (
    Persona(
        name="base_rate",
        system_prompt=(
            "You are the base_rate analyst. Start from the reference class and its "
            "historical frequency before reading any narrative. State the prior you "
            "used and only then adjust for case-specific evidence. Penalise yourself "
            "for relying on vivid but unrepresentative stories."
        ),
        weight=1.2,
    ),
    Persona(
        name="bull",
        system_prompt=(
            "You are the bull analyst. Build the strongest honest case for the event "
            "happening. Cite the documents that support it, but flag the single "
            "assumption most likely to break your case."
        ),
    ),
    Persona(
        name="bear",
        system_prompt=(
            "You are the bear analyst. Build the strongest honest case against the "
            "event. Look for base-rate neglect, survivorship bias and incentives of "
            "the sources. Do not simply invert the bull case; add new information."
        ),
    ),
)

RED_TEAM = Persona(
    name="red_team",
    system_prompt=(
        "You are the red_team reviewer. Given a panel probability and its reasoning, "
        "identify what the panel collectively missed: correlated evidence, shared "
        "source bias, ambiguous resolution criteria. Recommend a probability shift "
        "between -0.3 and 0.3 and justify it. If the panel is well calibrated for "
        "the evidence available, recommend zero."
    ),
)


def build_analyst_prompt(question: Question, evidence: list[Evidence], persona: str) -> str:
    lines = [
        f"analyst_persona={persona}",
        f"question: {question.text}",
        f"resolution_criteria: {question.resolution_criteria}",
        f"resolve_by: {question.resolve_by.isoformat()}",
        f"reference_class: {question.reference_class or 'not specified'}",
        "",
        "retrieved documents (newest weighted higher):",
    ]
    for doc in evidence:
        lines.append(
            f"- doc_id={doc.doc_id} source={doc.source} "
            f"published={doc.published_at.date()} score={doc.score:.3f}\n  {doc.text}"
        )
    lines += [
        "",
        "Respond with JSON: "
        '{"persona": str, "probability": float 0..1, "reasoning": str, '
        '"evidence_ids": [str], "confidence": float 0..1}.',
    ]
    return "\n".join(lines)


def build_red_team_prompt(question: Question, views: list[AnalystView], mean: float) -> str:
    reasoning = "\n".join(f"- {v.persona}: {v.probability:.3f} — {v.reasoning}" for v in views)
    return (
        f"question: {question.text}\n"
        f"resolution_criteria: {question.resolution_criteria}\n"
        f"panel_mean={mean:.4f}\n\n"
        f"panel reasoning:\n{reasoning}\n\n"
        'Respond with JSON: {"overlooked": [str], "probability_shift": float, "notes": str}.'
    )


def aggregate(
    views: list[AnalystView],
    critique: Critique | None = None,
) -> tuple[float, float, float, float]:
    """Return (probability, lower, upper, disagreement).

    Confidence-weighted mean, trimmed when the panel is large enough for one
    outlier to be disposable, then shifted by the red team. The interval is
    driven by panel disagreement rather than a fixed constant: when the panel
    agrees for correlated reasons the red team shift is what widens it.
    """
    if not views:
        raise ValueError("cannot aggregate an empty panel")

    probs = [v.probability for v in views]
    weights = [max(v.confidence, 0.05) for v in views]

    if len(views) >= 4:
        ordered = sorted(zip(probs, weights, strict=True))
        ordered = ordered[1:-1]
        probs_w = ordered
    else:
        probs_w = list(zip(probs, weights, strict=True))

    total_weight = sum(w for _, w in probs_w)
    mean = sum(p * w for p, w in probs_w) / total_weight

    disagreement = statistics.pstdev(probs) if len(probs) > 1 else 0.0

    shift = critique.probability_shift if critique else 0.0
    probability = min(0.98, max(0.02, mean + shift))

    half_width = min(0.35, max(0.05, 1.5 * disagreement + 0.06 + abs(shift)))
    lower = max(0.01, probability - half_width)
    upper = min(0.99, probability + half_width)
    return round(probability, 4), round(lower, 4), round(upper, 4), round(disagreement, 4)
