"""
Tests for the FSRS-5 scheduler: curves, recurrences, solver, forecast (v2).
"""
import sys
from pathlib import Path

_service_root = Path(__file__).resolve().parent.parent
if str(_service_root) not in sys.path:
    sys.path.insert(0, str(_service_root))

import pytest
from common.contracts import StudentConceptMastery

from fsrs_scheduler.fsrs5_optimizer import FSRS5Scheduler


@pytest.fixture
def sched():
    return FSRS5Scheduler()


def _state(stability=5.0, difficulty=5.0, due=None):
    return StudentConceptMastery(
        student_id="s1", concept_id="c1", bkt_p_mastery=0.5,
        bkt_p_slip=0.1, bkt_p_guess=0.2, bkt_p_transit=0.15,
        fsrs_stability=stability, fsrs_difficulty=difficulty, next_review_due=due,
    )


def test_retrievability_at_zero_days_is_one(sched):
    assert sched.retrievability(0, 10.0) == 1.0


def test_retrievability_decreases_with_time(sched):
    r1 = sched.retrievability(1, 10.0)
    r2 = sched.retrievability(10, 10.0)
    r3 = sched.retrievability(40, 10.0)
    assert 1.0 > r1 > r2 > r3 > 0.0


def test_retrievability_at_stability_point(sched):
    # R(S, S) = (1 + 19/81) ** -0.5 ≈ 0.8997
    assert sched.retrievability(10.0, 10.0) == pytest.approx(0.8997, abs=0.01)


def test_interval_inverts_retrievability(sched):
    interval = sched.interval_for_retention(20.0, 0.90)
    assert sched.retrievability(interval, 20.0) == pytest.approx(0.90, abs=0.01)


def test_initial_stability_increases_with_grade(sched):
    values = [sched.initial_stability(g) for g in (1, 2, 3, 4)]
    assert values == sorted(values)


def test_initial_difficulty_decreases_with_grade(sched):
    values = [sched.initial_difficulty(g) for g in (1, 2, 3, 4)]
    assert values == sorted(values, reverse=True)
    assert all(1.0 <= v <= 10.0 for v in values)


def test_update_difficulty_direction(sched):
    base = 5.0
    assert sched.update_difficulty(base, 1) > base   # Again -> harder
    assert sched.update_difficulty(base, 4) < base   # Easy -> easier
    assert sched.update_difficulty(base, 3) == pytest.approx(base, abs=0.6)


def test_stability_after_success_grows_with_grade(sched):
    s = 10.0
    hard = sched.stability_after_success(s, 5.0, 0.9, 2)
    good = sched.stability_after_success(s, 5.0, 0.9, 3)
    easy = sched.stability_after_success(s, 5.0, 0.9, 4)
    assert s < hard < good < easy


def test_stability_after_lapse_contracts(sched):
    assert sched.stability_after_lapse(10.0, 5.0, 0.9) < 10.0


def test_review_card_first_review(sched):
    updated = sched.review_card(_state(), rating=3, days_elapsed=0, is_first_review=True)
    assert updated.fsrs_stability == pytest.approx(sched.initial_stability(3), abs=0.01)
    assert updated.next_review_due is not None


def test_review_card_success_increases_stability(sched):
    state = _state(stability=8.0)
    updated = sched.review_card(state, rating=3, days_elapsed=7)
    assert updated.fsrs_stability > 8.0
    lapse = sched.review_card(state, rating=1, days_elapsed=7)
    assert lapse.fsrs_stability < 8.0


def test_review_card_invalid_rating_raises(sched):
    with pytest.raises(ValueError):
        sched.review_card(_state(), rating=7, days_elapsed=1)


def test_weights_validation():
    with pytest.raises(ValueError):
        FSRS5Scheduler(weights=(0.1, 0.2))


def test_forecast_workload_shape_and_due_today(sched):
    due_state = _state(due="2000-01-01T00:00:00+00:00")  # long overdue
    forecast = sched.forecast_workload([due_state], horizon_days=14)
    assert len(forecast) == 14
    assert forecast[0]["due_count"] >= 1
    cumulative = [f["cumulative"] for f in forecast]
    assert cumulative == sorted(cumulative)


def test_solve_optimal_retention_returns_grid_value(sched):
    result = sched.solve_optimal_retention(
        stabilities=[3.0, 8.0, 15.0, 30.0], horizon_days=90
    )
    assert 0.70 <= result["optimal_retention"] <= 0.98
    assert result["expected_reviews"] > 0
    assert result["optimal_retention"] in result["workload_by_retention"]
