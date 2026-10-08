import numpy as np

from logos.evaluation.metrics import calibrate_threshold, fpr_at_threshold, tpr_at_threshold


def test_tpr_counts_machine_at_or_above_threshold():
    scores = np.array([0.2, 0.4, 0.6, 0.8])
    labels = np.ones(4)
    assert tpr_at_threshold(scores, labels, tau=0.5) == 0.5


def test_fpr_counts_human_at_or_above_threshold():
    scores = np.array([0.2, 0.4, 0.6, 0.8])
    labels = np.zeros(4)
    assert fpr_at_threshold(scores, labels, tau=0.5) == 0.5


def test_calibrated_threshold_holds_the_fpr_budget():
    scores = np.arange(100.0)
    labels = np.zeros(100)
    tau = calibrate_threshold(scores, labels, target_fpr=0.01)
    assert fpr_at_threshold(scores, labels, tau) <= 0.01


def test_separable_data_catches_all_machine_at_1pct_fpr():
    human = np.linspace(0, 1, 1000)
    machine = np.linspace(1, 2, 1000)
    scores = np.concatenate([human, machine])
    labels = np.concatenate([np.zeros(1000), np.ones(1000)])
    tau = calibrate_threshold(scores, labels, target_fpr=0.01)
    assert tpr_at_threshold(scores, labels, tau) == 1.0


def test_stricter_budget_needs_a_higher_threshold():
    scores = np.arange(1000.0)
    labels = np.zeros(1000)
    assert calibrate_threshold(scores, labels, 0.001) >= calibrate_threshold(scores, labels, 0.01)


def test_fpr_budget_holds_when_budget_is_not_a_whole_number_of_texts():
    rng = np.random.default_rng(0)
    scores = rng.random(7639)
    labels = np.zeros(7639)
    tau = calibrate_threshold(scores, labels, target_fpr=0.01)
    assert fpr_at_threshold(scores, labels, tau) <= 0.01


def test_fpr_budget_holds_with_tied_scores():
    scores = np.array([0.0] * 95 + [0.9] * 5)
    labels = np.zeros(100)
    tau = calibrate_threshold(scores, labels, target_fpr=0.01)
    assert fpr_at_threshold(scores, labels, tau) <= 0.01