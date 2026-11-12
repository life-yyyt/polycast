from polycast.calibration import PlattScaler
from polycast.evaluation import recalibrate, split


def test_split_shuffles_before_splitting():
    # An ordered file would put every outcome=1 in the training set and every
    # outcome=0 in the test set, which is exactly the case that breaks a
    # fitted recalibration.
    pairs = [(0.9, 1)] * 10 + [(0.1, 0)] * 10
    train, test = split(pairs, train_fraction=0.7)
    assert len(train) == 14
    assert len(test) == 6
    assert {y for _, y in train} == {0, 1}
    assert {y for _, y in test} == {0, 1}


def test_split_is_reproducible():
    pairs = [(0.9, 1)] * 10 + [(0.1, 0)] * 10
    assert split(pairs) == split(pairs)


def test_platt_scaler_refuses_a_single_class_training_set():
    scaler = PlattScaler().fit([(0.9, 1)] * 12, lr=0.5, epochs=3000)
    assert scaler.a == 1.0 and scaler.b == 0.0


def test_recalibrate_returns_held_out_scores():
    pairs = [(0.7, 1)] * 10 + [(0.3, 0)] * 10
    result = recalibrate(pairs, train_fraction=0.8)
    assert result["train"] is not None
    assert result["test_raw"] is not None
    assert result["test_recalibrated"] is not None
    assert "a" in result["scaler"] and "b" in result["scaler"]
