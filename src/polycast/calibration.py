"""Scoring and recalibration.

Two rules of thumb behind this module:

* Brier score alone is not enough. A forecaster can post a decent Brier while
  being systematically overconfident, which is the failure mode that actually
  hurts a downstream consumer sizing positions on these numbers.
* Recalibration is fitted on held-out resolutions only. Fitting Platt scaling on
  the same questions you then score is a quiet way to manufacture skill.
"""

from __future__ import annotations

import math

from .schemas import CalibrationBin, CalibrationReport

EPS = 1e-6


def clip(p: float) -> float:
    return min(1.0 - EPS, max(EPS, p))


def brier_score(pairs: list[tuple[float, int]]) -> float:
    if not pairs:
        raise ValueError("no forecasts to score")
    # No clipping here: a perfect 0/1 forecast must score exactly zero, and
    # probabilities are already constrained to [0, 1] upstream.
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def log_loss(pairs: list[tuple[float, int]]) -> float:
    if not pairs:
        raise ValueError("no forecasts to score")
    total = 0.0
    for p, y in pairs:
        p = clip(p)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(pairs)


def reliability_bins(pairs: list[tuple[float, int]], n_bins: int = 10) -> list[CalibrationBin]:
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        members = [(p, y) for p, y in pairs if lo <= p < hi or (i == n_bins - 1 and p == 1.0)]
        if not members:
            continue
        bins.append(
            CalibrationBin(
                lower=lo,
                upper=hi,
                count=len(members),
                mean_predicted=sum(p for p, _ in members) / len(members),
                observed_frequency=sum(y for _, y in members) / len(members),
            )
        )
    return bins


def evaluate(pairs: list[tuple[float, int]], n_bins: int = 10) -> CalibrationReport:
    if not pairs:
        raise ValueError("no forecasts to score")
    base_rate = sum(y for _, y in pairs) / len(pairs)
    return CalibrationReport(
        n=len(pairs),
        brier=brier_score(pairs),
        log_loss=log_loss(pairs),
        base_rate_brier=brier_score([(base_rate, y) for _, y in pairs]),
        bins=reliability_bins(pairs, n_bins),
    )


def _logit(p: float) -> float:
    p = clip(p)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)
    return z / (1 + z)


class PlattScaler:
    """Logistic recalibration on the log-odds of the raw probability.

    Deliberately fitted on logit(p) rather than p: overconfident forecasters
    cluster near 0 and 1, and the interesting correction happens out there.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0) -> None:
        self.a, self.b = a, b

    def fit(self, pairs: list[tuple[float, int]], lr: float = 0.1, epochs: int = 2000) -> "PlattScaler":
        if len(pairs) < 8:
            # Below this the fit mostly memorises noise; better to ship the raw
            # probabilities and say so than to pretend we calibrated.
            return self
        outcomes = {y for _, y in pairs}
        if len(outcomes) < 2:
            # A single-class training set drives the intercept to infinity and
            # the fitted curve extrapolates catastrophically on the test set.
            return self
        xs = [_logit(p) for p, _ in pairs]
        ys = [y for _, y in pairs]
        a, b = self.a, self.b
        n = len(xs)
        for _ in range(epochs):
            ga = gb = 0.0
            for x, y in zip(xs, ys):
                err = _sigmoid(a * x + b) - y
                ga += err * x
                gb += err
            a -= lr * ga / n
            b -= lr * gb / n
        self.a, self.b = a, b
        return self

    def transform(self, p: float) -> float:
        return clip(_sigmoid(self.a * _logit(p) + self.b))
