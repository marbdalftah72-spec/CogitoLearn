"""
BKT parameter fitting, forgetting decay and prediction evaluation.
==================================================================
Advanced companion to :class:`BKTUpdater` (v1).  BKT is a two-state
hidden Markov model per concept, so its four parameters
``(P(L0), P(T), P(G), P(S))`` can be *learned from data* instead of being
hand-set:

* **Grid search** — exhaustive search over a coarse grid maximising the
  exact sequence log-likelihood (forward algorithm).
* **Simplified EM (Baum–Welch)** — forward/backward recursions producing
  posterior state occupancies ``gamma`` and transition counts ``xi``,
  followed by closed-form M-steps for all four parameters.

Also provides:

* **Forgetting decay** — exponential decay of mastery between sessions
  with a configurable half-life, and
* **Prediction evaluation** — one-step-ahead filtered predictions scored
  with AUC, RMSE, accuracy and log-loss.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from common.contracts import StudentConceptMastery

from config import BKT_FIT_GRID, BKT_FORGETTING_HALF_LIFE_DAYS

logger = logging.getLogger(__name__)

_EPS = 1e-9


@dataclass(frozen=True)
class BKTParameters:
    """The four BKT parameters for one concept."""

    p_l0: float = 0.15
    p_transit: float = 0.15
    p_guess: float = 0.20
    p_slip: float = 0.10

    def validate(self) -> "BKTParameters":
        """Ensure every parameter is a valid probability."""
        for name, value in (
            ("p_l0", self.p_l0),
            ("p_transit", self.p_transit),
            ("p_guess", self.p_guess),
            ("p_slip", self.p_slip),
        ):
            if not (0.0 < value < 1.0):
                raise ValueError(f"{name} must be in (0, 1), got {value}")
        return self

    def as_state_update(self) -> Dict[str, float]:
        """Return a dict suitable for ``StudentConceptMastery.model_copy``."""
        return {
            "bkt_p_slip": round(self.p_slip, 4),
            "bkt_p_guess": round(self.p_guess, 4),
            "bkt_p_transit": round(self.p_transit, 4),
        }


def _emission(params: BKTParameters, learned: bool, correct: bool) -> float:
    """P(observation | hidden state)."""
    if learned:
        return (1.0 - params.p_slip) if correct else params.p_slip
    return params.p_guess if correct else (1.0 - params.p_guess)


def _forward(
    sequence: Sequence[bool], params: BKTParameters
) -> tuple[List[List[float]], List[float]]:
    """Scaled forward recursion. Returns (alpha, scale) with loglik = sum(log(scale))."""
    alpha: List[List[float]] = []
    scales: List[float] = []
    # t = 0 prior
    a0 = [
        (1.0 - params.p_l0) * _emission(params, False, sequence[0]),
        params.p_l0 * _emission(params, True, sequence[0]),
    ]
    s = sum(a0) or _EPS
    alpha.append([a0[0] / s, a0[1] / s])
    scales.append(s)
    for t in range(1, len(sequence)):
        prev = alpha[-1]
        # transition: 0 -> {0: 1-pT, 1: pT}; 1 -> {0: 0, 1: 1}
        trans0 = prev[0] * (1.0 - params.p_transit)
        trans1 = prev[0] * params.p_transit + prev[1]
        a = [
            trans0 * _emission(params, False, sequence[t]),
            trans1 * _emission(params, True, sequence[t]),
        ]
        s = sum(a) or _EPS
        alpha.append([a[0] / s, a[1] / s])
        scales.append(s)
    return alpha, scales


def _backward(sequence: Sequence[bool], params: BKTParameters, scales: List[float]) -> List[List[float]]:
    """Scaled backward recursion consistent with :func:`_forward` scaling."""
    n = len(sequence)
    beta: List[List[float]] = [[1.0, 1.0] for _ in range(n)]
    for t in range(n - 2, -1, -1):
        nxt = sequence[t + 1]
        b_next0 = beta[t + 1][0] * _emission(params, False, nxt)
        b_next1 = beta[t + 1][1] * _emission(params, True, nxt)
        beta[t][0] = (
            (1.0 - params.p_transit) * b_next0 + params.p_transit * b_next1
        ) / scales[t + 1]
        beta[t][1] = b_next1 / scales[t + 1]
    return beta


class BKTParameterFitter:
    """Learn BKT parameters from response sequences of a single concept."""

    def __init__(self, grid: Sequence[float] = BKT_FIT_GRID) -> None:
        self.grid = tuple(grid)

    # -- likelihood ---------------------------------------------------------
    def log_likelihood(
        self, sequences: Sequence[Sequence[bool]], params: BKTParameters
    ) -> float:
        """Total log-likelihood of all sequences under ``params``."""
        total = 0.0
        for seq in sequences:
            if not seq:
                continue
            _, scales = _forward(seq, params)
            total += sum(math.log(max(s, _EPS)) for s in scales)
        return total

    # -- grid search --------------------------------------------------------
    def fit_grid_search(
        self, sequences: Sequence[Sequence[bool]]
    ) -> BKTParameters:
        """Exhaustive coarse grid search maximising log-likelihood."""
        best = BKTParameters()
        best_ll = -math.inf
        for p_l0 in self.grid:
            for p_t in self.grid:
                for p_g in self.grid:
                    for p_s in self.grid:
                        cand = BKTParameters(p_l0, p_t, p_g, p_s)
                        ll = self.log_likelihood(sequences, cand)
                        if ll > best_ll:
                            best_ll, best = ll, cand
        logger.info("BKT grid search best LL=%.4f params=%s", best_ll, best)
        return best

    # -- EM (Baum–Welch, simplified closed-form M-step) ---------------------
    def fit_em(
        self,
        sequences: Sequence[Sequence[bool]],
        init: BKTParameters | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> BKTParameters:
        """Fit parameters with expectation-maximisation.

        E-step: forward/backward posteriors per sequence.
        M-step (closed form):
          * ``P(L0)``  = mean posterior P(L_0 = 1)
          * ``P(T)``   = E[# 0->1 transitions] / E[# visits to state 0 with a successor]
          * ``P(G)``   = E[# correct while unlearned] / E[# visits to state 0]
          * ``P(S)``   = E[# incorrect while learned] / E[# visits to state 1]
        """
        params = (init or BKTParameters()).validate()
        prev_ll = -math.inf
        for iteration in range(max_iter):
            ll = self.log_likelihood(sequences, params)
            if abs(ll - prev_ll) < tol:
                logger.info("BKT EM converged at iter %d (LL=%.4f)", iteration, ll)
                break
            prev_ll = ll

            sum_gamma0_learned = 0.0
            n_seq = 0
            xi_01 = 0.0
            gamma0_with_next = 0.0
            correct_in_0 = 0.0
            visits_0 = 0.0
            incorrect_in_1 = 0.0
            visits_1 = 0.0

            for seq in sequences:
                if not seq:
                    continue
                n_seq += 1
                alpha, scales = _forward(seq, params)
                beta = _backward(seq, params, scales)
                n = len(seq)
                gamma: List[List[float]] = []
                for t in range(n):
                    g0 = alpha[t][0] * beta[t][0]
                    g1 = alpha[t][1] * beta[t][1]
                    tot = (g0 + g1) or _EPS
                    gamma.append([g0 / tot, g1 / tot])
                sum_gamma0_learned += gamma[0][1]
                for t in range(n):
                    visits_0 += gamma[t][0]
                    visits_1 += gamma[t][1]
                    if seq[t]:
                        correct_in_0 += gamma[t][0]
                    else:
                        incorrect_in_1 += gamma[t][1]
                    if t < n - 1:
                        gamma0_with_next += gamma[t][0]
                        # xi_t(0->1) ∝ alpha_t(0) * pT * emit_{t+1}(1) * beta_{t+1}(1)
                        xi = (
                            alpha[t][0]
                            * params.p_transit
                            * _emission(params, True, seq[t + 1])
                            * beta[t + 1][1]
                            / scales[t + 1]
                        )
                        xi_01 += xi

            if n_seq == 0:
                break
            new = BKTParameters(
                p_l0=_clip(sum_gamma0_learned / n_seq),
                p_transit=_clip(xi_01 / gamma0_with_next) if gamma0_with_next > _EPS else params.p_transit,
                p_guess=_clip(correct_in_0 / visits_0) if visits_0 > _EPS else params.p_guess,
                p_slip=_clip(incorrect_in_1 / visits_1) if visits_1 > _EPS else params.p_slip,
            )
            params = new
        return params

    def fit(
        self,
        sequences: Sequence[Sequence[bool]],
        method: str = "em",
    ) -> BKTParameters:
        """Convenience dispatcher: ``method`` is ``"em"`` or ``"grid"``."""
        if method == "grid":
            return self.fit_grid_search(sequences)
        if method == "em":
            seed = self.fit_grid_search(sequences) if len(sequences) >= 3 else None
            return self.fit_em(sequences, init=seed)
        raise ValueError(f"Unknown fit method: {method!r}")


def _clip(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return float(max(lo, min(hi, p)))


# ---------------------------------------------------------------------------
# Forgetting decay
# ---------------------------------------------------------------------------
def apply_forgetting_decay(
    p_mastery: float,
    days_elapsed: float,
    half_life_days: float = BKT_FORGETTING_HALF_LIFE_DAYS,
    floor: float = 0.05,
) -> float:
    """Decay mastery exponentially toward ``floor`` between sessions.

    ``p' = floor + (p - floor) * 0.5 ** (days / half_life)``
    """
    if days_elapsed <= 0:
        return float(p_mastery)
    decayed = floor + (p_mastery - floor) * (0.5 ** (days_elapsed / half_life_days))
    return float(max(0.0001, min(0.9999, decayed)))


def decay_state(
    state: StudentConceptMastery,
    days_elapsed: float,
    half_life_days: float = BKT_FORGETTING_HALF_LIFE_DAYS,
) -> StudentConceptMastery:
    """Return a copy of ``state`` with forgetting decay applied to mastery."""
    new_p = apply_forgetting_decay(state.bkt_p_mastery, days_elapsed, half_life_days)
    return state.model_copy(update={"bkt_p_mastery": round(new_p, 4)})


# ---------------------------------------------------------------------------
# Prediction evaluation
# ---------------------------------------------------------------------------
@dataclass
class PredictionMetrics:
    """One-step-ahead prediction quality for a BKT parameterisation."""

    auc: float
    rmse: float
    accuracy: float
    log_loss: float
    n_predictions: int
    predictions: List[float] = field(default_factory=list)


class BKTPredictionEvaluator:
    """Score BKT parameters by filtered one-step-ahead predictions."""

    def __init__(self, params: BKTParameters) -> None:
        self.params = params.validate()

    def _predict_sequence(self, sequence: Sequence[bool]) -> List[float]:
        preds: List[float] = []
        p_learned = self.params.p_l0
        for obs in sequence:
            p_correct = p_learned * (1 - self.params.p_slip) + (1 - p_learned) * self.params.p_guess
            preds.append(p_correct)
            # posterior update given the observation, then learning transition
            if obs:
                num = p_learned * (1 - self.params.p_slip)
                den = num + (1 - p_learned) * self.params.p_guess
            else:
                num = p_learned * self.params.p_slip
                den = num + (1 - p_learned) * (1 - self.params.p_guess)
            posterior = num / den if den > _EPS else p_learned
            p_learned = posterior + (1 - posterior) * self.params.p_transit
        return preds

    def evaluate(self, sequences: Sequence[Sequence[bool]]) -> PredictionMetrics:
        """Compute AUC / RMSE / accuracy / log-loss over all sequences."""
        all_preds: List[float] = []
        all_obs: List[int] = []
        for seq in sequences:
            preds = self._predict_sequence(seq)
            all_preds.extend(preds)
            all_obs.extend(1 if o else 0 for o in seq)
        if not all_preds:
            return PredictionMetrics(0.5, 0.0, 0.0, 0.0, 0, [])
        n = len(all_preds)
        rmse = math.sqrt(sum((p - o) ** 2 for p, o in zip(all_preds, all_obs)) / n)
        accuracy = sum(1 for p, o in zip(all_preds, all_obs) if (p >= 0.5) == bool(o)) / n
        log_loss = -sum(
            o * math.log(max(p, _EPS)) + (1 - o) * math.log(max(1 - p, _EPS))
            for p, o in zip(all_preds, all_obs)
        ) / n
        auc = _auc(all_preds, all_obs)
        return PredictionMetrics(
            auc=round(auc, 4),
            rmse=round(rmse, 4),
            accuracy=round(accuracy, 4),
            log_loss=round(log_loss, 4),
            n_predictions=n,
            predictions=all_preds,
        )


def _auc(preds: List[float], obs: List[int]) -> float:
    """Rank-based AUC (Mann–Whitney U) with tie handling."""
    pos = [(p, o) for p, o in zip(preds, obs) if o == 1]
    neg = [(p, o) for p, o in zip(preds, obs) if o == 0]
    if not pos or not neg:
        return 0.5
    ranked = sorted(zip(preds, obs), key=lambda x: x[0])
    ranks: Dict[int, float] = {}
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(ranks[idx] for idx, (_, o) in enumerate(ranked) if o == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
