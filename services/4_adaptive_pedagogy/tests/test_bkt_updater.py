"""
Tests for Bayesian Knowledge Tracing (BKT) Updater.
"""
import sys
from pathlib import Path

_service_root = Path(__file__).resolve().parent.parent
if str(_service_root) not in sys.path:
    sys.path.insert(0, str(_service_root))

import pytest
from common.contracts import StudentConceptMastery
from common.exceptions.domain_errors import PedagogyStateError
from bkt_engine.bkt_updater import BKTUpdater


@pytest.fixture
def bkt():
    return BKTUpdater()


@pytest.fixture
def initial_state():
    return StudentConceptMastery(
        student_id="student_123",
        concept_id="concept_attention",
        bkt_p_mastery=0.15,
        bkt_p_slip=0.10,
        bkt_p_guess=0.20,
        bkt_p_transit=0.15,
        fsrs_stability=1.0,
        fsrs_difficulty=5.0,
    )


def test_predict_correct_probability(bkt, initial_state):
    # P(C) = P(L)*(1-S) + (1-P(L))*G = 0.15*0.90 + 0.85*0.20 = 0.135 + 0.170 = 0.305
    p_corr = bkt.predict_correct_probability(initial_state)
    assert pytest.approx(p_corr, 0.01) == 0.305


def test_update_mastery_correct_increases_probability(bkt, initial_state):
    updated = bkt.update_mastery(initial_state, is_correct=True)
    assert updated.bkt_p_mastery > initial_state.bkt_p_mastery
    assert updated.bkt_p_mastery > 0.50


def test_update_mastery_incorrect_handles_lapse(bkt, initial_state):
    updated = bkt.update_mastery(initial_state, is_correct=False)
    correct_updated = bkt.update_mastery(initial_state, is_correct=True)
    assert updated.bkt_p_mastery < correct_updated.bkt_p_mastery


def test_batch_update_monotonic_increase_on_all_correct(bkt, initial_state):
    updated = bkt.batch_update(initial_state, [True, True, True, True, True])
    assert updated.bkt_p_mastery > 0.90
    assert bkt.classify_mastery(updated.bkt_p_mastery) in ["LEARNING", "MASTERED"]


def test_classify_mastery(bkt):
    assert bkt.classify_mastery(0.96) == "MASTERED"
    assert bkt.classify_mastery(0.70) == "LEARNING"
    assert bkt.classify_mastery(0.40) == "UNLEARNED"


def test_invalid_probability_raises_domain_error(bkt, initial_state):
    bad_state = initial_state.model_copy(update={"bkt_p_mastery": 1.5})
    with pytest.raises(PedagogyStateError):
        bkt.update_mastery(bad_state, is_correct=True)
