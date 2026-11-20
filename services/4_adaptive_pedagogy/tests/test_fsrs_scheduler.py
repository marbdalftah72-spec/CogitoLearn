"""
Tests for FSRS-4.5 Spaced Repetition Memory Optimizer.
"""
import sys
from pathlib import Path

_service_root = Path(__file__).resolve().parent.parent
if str(_service_root) not in sys.path:
    sys.path.insert(0, str(_service_root))

from datetime import datetime, timezone, timedelta
import pytest
from common.contracts import StudentConceptMastery
from fsrs_scheduler.memory_optimizer import FSRSScheduler


@pytest.fixture
def scheduler():
    return FSRSScheduler(desired_retention=0.90)


@pytest.fixture
def sample_state():
    return StudentConceptMastery(
        student_id="std_fsrs",
        concept_id="c_transformer",
        fsrs_stability=2.0,
        fsrs_difficulty=5.0,
        next_review_due=None,
    )


def test_retrievability_decay(scheduler):
    # At t = 0, R = 1.0
    assert scheduler.calculate_retrievability(stability=2.0, days_elapsed=0.0) == 1.0
    # Over time, R decays monotonically
    r1 = scheduler.calculate_retrievability(stability=2.0, days_elapsed=1.0)
    r5 = scheduler.calculate_retrievability(stability=2.0, days_elapsed=5.0)
    assert 0.0 < r5 < r1 < 1.0


def test_schedule_review_success_increases_stability(scheduler, sample_state):
    # Rating 3 (Good)
    updated = scheduler.schedule_review(sample_state, rating=3)
    assert updated.fsrs_stability > sample_state.fsrs_stability
    assert updated.next_review_due is not None


def test_schedule_review_lapse_reduces_stability(scheduler, sample_state):
    # Rating 1 (Again)
    updated = scheduler.schedule_review(sample_state, rating=1)
    assert updated.fsrs_stability < sample_state.fsrs_stability
    assert updated.fsrs_difficulty > sample_state.fsrs_difficulty


def test_get_due_concepts(scheduler, sample_state):
    # Never reviewed is due immediately
    assert sample_state in scheduler.get_due_concepts([sample_state])

    # Future date is not due
    now = datetime.now(timezone.utc)
    future_state = sample_state.model_copy(
        update={"next_review_due": (now + timedelta(days=5)).isoformat()}
    )
    assert future_state not in scheduler.get_due_concepts([future_state])

    # Past date is due
    past_state = sample_state.model_copy(
        update={"next_review_due": (now - timedelta(days=1)).isoformat()}
    )
    assert past_state in scheduler.get_due_concepts([past_state])
