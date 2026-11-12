"""Evaluation harness.

The number that matters is not the Brier score on the questions you developed
the system on. It is the score on resolved questions the system has never
seen, with any recalibration fitted on a disjoint split.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .calibration import PlattScaler, evaluate
from .pipeline import ForecastPipeline
from .schemas import ResolvedQuestion


def load_resolved(path: str | Path) -> list[ResolvedQuestion]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ResolvedQuestion(**item) for item in raw]


async def score_resolved(
    pipeline: ForecastPipeline, resolved: list[ResolvedQuestion]
) -> list[tuple[float, int]]:
    pairs: list[tuple[float, int]] = []
    for item in resolved:
        forecast = await pipeline.run(item.question)
        pairs.append((forecast.probability, item.outcome))
    return pairs


def split(
    pairs: list[tuple[float, int]], train_fraction: float = 0.7, seed: int = 1234
) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
    """Shuffle before splitting.

    A sequential split silently produces a pathological training set whenever
    the file happens to be ordered by outcome, and the recalibration fitted on
    it then extrapolates hard into the test set. Shuffling with a fixed seed
    keeps the split reproducible.
    """
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    cut = max(1, int(len(pairs) * train_fraction))
    train = [pairs[i] for i in order[:cut]]
    test = [pairs[i] for i in order[cut:]]
    return train, test


def recalibrate(
    pairs: list[tuple[float, int]], train_fraction: float = 0.7
) -> dict[str, Any]:
    """Fit Platt scaling on a training split, score on the held-out split."""
    train, test = split(pairs, train_fraction)
    if not test:
        return {
            "train": evaluate(train).model_dump() if train else None,
            "test": None,
            "scaler": None,
        }
    scaler = PlattScaler().fit(train)
    recalibrated = [(scaler.transform(p), y) for p, y in test]
    return {
        "train": evaluate(train).model_dump(),
        "test_raw": evaluate(test).model_dump(),
        "test_recalibrated": evaluate(recalibrated).model_dump(),
        "scaler": {"a": round(scaler.a, 4), "b": round(scaler.b, 4)},
    }
