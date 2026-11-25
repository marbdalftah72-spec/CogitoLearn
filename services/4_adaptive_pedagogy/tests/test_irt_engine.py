"""
Tests for the 2PL IRT engine: ability estimation, calibration, adaptive selection (v2).
"""
import sys
from pathlib import Path

_service_root = Path(__file__).resolve().parent.parent
if str(_service_root) not in sys.path:
    sys.path.insert(0, str(_service_root))

import random

import pytest

from irt_engine.irt_model import (
    AbilityEstimator,
    AdaptiveItemSelector,
    IRTItem,
    ItemCalibrator,
    fisher_information,
    prob_correct,
)


def _items(n=8, a=1.2):
    return [
        IRTItem(item_id=f"it{i}", concept_id="c1", discrimination=a, difficulty=-1.5 + 0.4 * i)
        for i in range(n)
    ]


def test_prob_correct_at_difficulty_is_half():
    assert prob_correct(0.3, 1.5, 0.3) == pytest.approx(0.5)


def test_prob_correct_monotonic_in_theta():
    probs = [prob_correct(t, 1.0, 0.0) for t in (-2, -1, 0, 1, 2)]
    assert probs == sorted(probs)
    assert probs[0] < 0.2 < probs[-1]


def test_fisher_information_peaks_near_difficulty():
    item = IRTItem("i1", "c1", discrimination=1.5, difficulty=0.5)
    near = fisher_information(0.5, item)
    far = fisher_information(3.0, item)
    assert near > far > 0


def test_estimate_all_correct_gives_positive_theta():
    est = AbilityEstimator().estimate(_items(), [True] * 8)
    assert est.theta > 0.5
    assert est.n_items == 8


def test_estimate_all_wrong_gives_negative_theta():
    est = AbilityEstimator().estimate(_items(), [False] * 8)
    assert est.theta < -0.5


def test_estimate_mixed_responses_near_zero():
    items = [IRTItem(f"i{i}", "c1", 1.2, 0.0) for i in range(10)]
    responses = [True, False] * 5
    est = AbilityEstimator().estimate(items, responses)
    assert abs(est.theta) < 0.6


def test_estimate_standard_error_shrinks_with_more_items():
    few = AbilityEstimator().estimate(_items(4), [True, False, True, False])
    many = AbilityEstimator().estimate(_items(16), [True, False] * 8)
    assert many.standard_error < few.standard_error


def test_estimate_length_mismatch_raises():
    with pytest.raises(ValueError):
        AbilityEstimator().estimate(_items(3), [True])


def test_estimate_empty_returns_prior():
    est = AbilityEstimator().estimate([], [])
    assert est.theta == 0.0 and est.n_items == 0


def _synthetic_calibration_data(seed=11):
    rng = random.Random(seed)
    difficulties = [-1.5, -0.75, 0.0, 0.75, 1.5]
    items = [IRTItem(f"q{i}", "c1", 1.0, 0.0) for i in range(5)]  # uncalibrated start
    true_thetas = {"weak": -1.5, "mid": 0.0, "strong": 1.5}
    students, matrix = [], []
    for sid, theta in true_thetas.items():
        for rep in range(12):
            students.append(f"{sid}_{rep}")
            row = []
            for b in difficulties:
                p = 1.0 / (1.0 + pow(2.718281828, -(theta - b)))
                row.append(rng.random() < p)
            matrix.append(row)
    return students, items, matrix, difficulties, true_thetas


def test_calibration_recovers_difficulty_ordering():
    students, items, matrix, difficulties, _ = _synthetic_calibration_data()
    result = ItemCalibrator(epochs=120).calibrate(students, items, matrix)
    fitted_b = [it.difficulty for it in result.items]
    # Easiest item (true b=-1.5) must end up easier than hardest (true b=1.5).
    assert fitted_b[0] < fitted_b[-1]
    assert fitted_b[0] < fitted_b[2] < fitted_b[-1] or fitted_b[0] < fitted_b[-1]


def test_calibration_recovers_theta_ordering():
    students, items, matrix, _, _ = _synthetic_calibration_data()
    result = ItemCalibrator(epochs=120).calibrate(students, items, matrix)
    weak = [t for s, t in result.thetas.items() if s.startswith("weak")]
    strong = [t for s, t in result.thetas.items() if s.startswith("strong")]
    assert sum(strong) / len(strong) > sum(weak) / len(weak)


def test_calibration_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ItemCalibrator().calibrate(["s1"], _items(3), [[True, False]])


def test_selector_picks_most_informative_item():
    selector = AdaptiveItemSelector()
    bank = [
        IRTItem("easy", "c1", 1.2, -2.0),
        IRTItem("match", "c1", 1.2, 0.8),
        IRTItem("hard", "c1", 1.2, 2.5),
    ]
    chosen = selector.select_next(0.8, bank)
    assert chosen is not None and chosen.item_id == "match"


def test_selector_skips_used_and_exhausts():
    selector = AdaptiveItemSelector()
    bank = [IRTItem("only", "c1", 1.0, 0.0)]
    assert selector.select_next(0.0, bank, used_item_ids={"only"}) is None


def test_adaptive_session_estimates_simulated_student():
    rng = random.Random(3)
    true_theta = 1.0
    bank = [
        IRTItem(f"b{i}", "c1", 1.3, -2.0 + 0.35 * i) for i in range(12)
    ]

    def respond(item):
        return rng.random() < prob_correct(true_theta, item.discrimination, item.difficulty)

    selector = AdaptiveItemSelector()
    estimate, administered = selector.run_adaptive_session(bank, respond, max_items=10)
    assert 0 < len(administered) <= 10
    assert len(set(administered)) == len(administered)  # no repeats
    assert abs(estimate.theta - true_theta) < 1.0
