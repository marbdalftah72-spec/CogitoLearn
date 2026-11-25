"""
Tests for BKT parameter fitting, forgetting decay and prediction evaluation (v2).
"""
import sys
from pathlib import Path

_service_root = Path(__file__).resolve().parent.parent
if str(_service_root) not in sys.path:
    sys.path.insert(0, str(_service_root))

import random

import pytest
from common.contracts import StudentConceptMastery

from bkt_engine.bkt_fitter import (
    BKTParameterFitter,
    BKTParameters,
    BKTPredictionEvaluator,
    apply_forgetting_decay,
    decay_state,
)


def _simulate(params: BKTParameters, n_students: int = 25, length: int = 14, seed: int = 7):
    """Simulate response sequences from a known BKT ground truth."""
    rng = random.Random(seed)
    sequences = []
    for _ in range(n_students):
        learned = rng.random() < params.p_l0
        seq = []
        for _ in range(length):
            if learned:
                seq.append(rng.random() > params.p_slip)
            else:
                seq.append(rng.random() < params.p_guess)
            if not learned and rng.random() < params.p_transit:
                learned = True
        sequences.append(seq)
    return sequences


@pytest.fixture
def ground_truth():
    return BKTParameters(p_l0=0.25, p_transit=0.25, p_guess=0.15, p_slip=0.08)


def test_log_likelihood_is_finite_and_negative(ground_truth):
    fitter = BKTParameterFitter()
    seqs = _simulate(ground_truth)
    ll = fitter.log_likelihood(seqs, ground_truth)
    assert ll < 0 and ll == ll  # finite, not NaN


def test_grid_search_beats_default_likelihood(ground_truth):
    fitter = BKTParameterFitter()
    seqs = _simulate(ground_truth)
    fitted = fitter.fit_grid_search(seqs)
    assert fitter.log_likelihood(seqs, fitted) >= fitter.log_likelihood(seqs, BKTParameters())


def test_em_improves_likelihood(ground_truth):
    fitter = BKTParameterFitter()
    seqs = _simulate(ground_truth)
    init = BKTParameters()
    fitted = fitter.fit_em(seqs, init=init)
    assert fitter.log_likelihood(seqs, fitted) >= fitter.log_likelihood(seqs, init)


def test_em_recovers_plausible_parameters(ground_truth):
    fitter = BKTParameterFitter()
    seqs = _simulate(ground_truth, n_students=40, length=16)
    fitted = fitter.fit_em(seqs, init=BKTParameters())
    # Fitted guess/slip should land in a plausible band around ground truth.
    assert abs(fitted.p_guess - ground_truth.p_guess) < 0.20
    assert abs(fitted.p_slip - ground_truth.p_slip) < 0.20
    assert 0.0 < fitted.p_transit < 1.0


def test_fit_dispatcher_rejects_unknown_method(ground_truth):
    fitter = BKTParameterFitter()
    with pytest.raises(ValueError):
        fitter.fit(_simulate(ground_truth, n_students=2), method="magic")


def test_parameters_validation():
    with pytest.raises(ValueError):
        BKTParameters(p_guess=1.5).validate()


def test_forgetting_decay_zero_days_is_identity():
    assert apply_forgetting_decay(0.8, 0) == 0.8


def test_forgetting_decay_half_life():
    # After exactly one half-life, mastery sits halfway to the floor.
    decayed = apply_forgetting_decay(0.9, 21.0, half_life_days=21.0, floor=0.05)
    assert decayed == pytest.approx(0.05 + (0.9 - 0.05) * 0.5, abs=1e-6)


def test_forgetting_decay_long_gap_approaches_floor():
    decayed = apply_forgetting_decay(0.95, 365.0, half_life_days=21.0, floor=0.05)
    assert decayed < 0.10


def test_decay_state_returns_updated_copy():
    state = StudentConceptMastery(
        student_id="s1", concept_id="c1", bkt_p_mastery=0.9,
        bkt_p_slip=0.1, bkt_p_guess=0.2, bkt_p_transit=0.15,
        fsrs_stability=1.0, fsrs_difficulty=5.0,
    )
    decayed = decay_state(state, days_elapsed=42.0)
    assert decayed.bkt_p_mastery < state.bkt_p_mastery
    assert state.bkt_p_mastery == 0.9  # original untouched (immutable style)


def test_prediction_evaluator_metrics_on_synthetic(ground_truth):
    seqs = _simulate(ground_truth, n_students=30, length=12)
    metrics = BKTPredictionEvaluator(ground_truth).evaluate(seqs)
    assert metrics.n_predictions == 30 * 12
    assert 0.0 <= metrics.rmse <= 1.0
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.auc > 0.55  # better than chance with true parameters
    assert metrics.log_loss > 0


def test_prediction_evaluator_empty_input():
    metrics = BKTPredictionEvaluator(BKTParameters()).evaluate([])
    assert metrics.n_predictions == 0
    assert metrics.auc == 0.5
