import math

from polycast.calibration import (
    PlattScaler,
    brier_score,
    evaluate,
    log_loss,
    reliability_bins,
)


def test_brier_of_perfect_and_worst_forecasts():
    assert brier_score([(1.0, 1), (0.0, 0)]) == 0.0
    assert brier_score([(1.0, 0), (0.0, 1)]) == 1.0


def test_brier_of_base_rate_forecast():
    pairs = [(0.5, 1), (0.5, 0), (0.5, 1)]
    assert math.isclose(brier_score(pairs), 0.25)


def test_log_loss_penalises_confident_errors():
    wrong_confident = log_loss([(0.99, 0)])
    wrong_hedged = log_loss([(0.6, 0)])
    assert wrong_confident > wrong_hedged


def test_reliability_bins_group_by_probability():
    pairs = [(0.05, 0), (0.85, 1)]
    bins = reliability_bins(pairs, n_bins=10)
    assert len(bins) == 2
    assert bins[0].count == 1
    assert bins[0].observed_frequency == 0.0
    assert bins[1].count == 1
    assert bins[1].observed_frequency == 1.0


def test_evaluate_reports_skill_against_base_rate():
    pairs = [(0.9, 1)] * 6 + [(0.2, 0)] * 6
    report = evaluate(pairs)
    assert report.n == 12
    assert report.brier < report.base_rate_brier
    assert report.skill_vs_base_rate > 0


def test_platt_scaler_steepens_slope_when_forecasts_are_underconfident():
    # Predicted 0.6 / 0.4 against a realised 70% / 30%: the raw probabilities
    # are underconfident, so the fitted slope should move above 1.
    pairs = [(0.6, 1)] * 7 + [(0.6, 0)] * 3 + [(0.4, 1)] * 3 + [(0.4, 0)] * 7
    scaler = PlattScaler().fit(pairs, lr=0.5, epochs=4000)
    assert scaler.a > 1.1
    assert scaler.transform(0.6) > 0.6


def test_platt_scaler_cannot_shrink_symmetric_confident_forecasts():
    # A 0.95 / 0.05 pair against a 50/50 base rate is overconfident, but Platt
    # scaling is a monotone transform of the log-odds and cannot pull both ends
    # inward without making the log loss worse. Beta or isotonic
    # recalibration is the tool for that failure mode, not this one.
    pairs = [(0.95, 1)] * 10 + [(0.05, 0)] * 10
    scaler = PlattScaler().fit(pairs, lr=0.5, epochs=3000)
    assert scaler.transform(0.95) >= 0.9


def test_platt_scaler_ignores_tiny_samples():
    scaler = PlattScaler().fit([(0.9, 1), (0.1, 0)])
    assert scaler.a == 1.0 and scaler.b == 0.0
