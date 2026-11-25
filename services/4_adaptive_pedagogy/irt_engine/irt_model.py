"""
Item Response Theory (IRT) engine — 2PL model.
===============================================
Implements the two-parameter logistic (2PL) model:

    P(correct | theta, a, b) = 1 / (1 + exp(-a * (theta - b)))

where ``theta`` is the student's latent ability, ``a`` the item
discrimination and ``b`` the item difficulty.

Capabilities
------------
* **Ability estimation** — maximum-likelihood estimation of ``theta``
  via Newton–Raphson on the log-likelihood (with a standard-normal prior
  fallback MAP step when the likelihood is flat, e.g. all-correct data).
* **Item calibration** — joint maximum-likelihood calibration of ``(a, b)``
  for every item from a full students x items response matrix, using
  alternating optimisation (fix thetas, update items by gradient ascent;
  fix items, re-estimate thetas), with L2 shrinkage on ``a``.
* **Adaptive item selection** — Fisher information of an item at ``theta``
  is ``I(theta) = a^2 * P * (1 - P)``; the next item is the unused item
  maximising information at the current ability estimate.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from config import (
    IRT_CALIBRATION_EPOCHS,
    IRT_CALIBRATION_L2,
    IRT_CALIBRATION_LR,
    IRT_NEWTON_MAX_ITER,
    IRT_NEWTON_TOL,
    IRT_THETA_MAX,
    IRT_THETA_MIN,
    IRT_THETA_SE_FLOOR,
)

logger = logging.getLogger(__name__)

_EPS = 1e-9


@dataclass
class IRTItem:
    """A calibrated (or to-be-calibrated) 2PL item."""

    item_id: str
    concept_id: str
    discrimination: float = 1.0  # a
    difficulty: float = 0.0      # b

    def validate(self) -> "IRTItem":
        if self.discrimination <= 0:
            raise ValueError("discrimination (a) must be positive")
        return self


@dataclass
class AbilityEstimate:
    """Result of a theta estimation."""

    theta: float
    standard_error: float
    n_items: int
    log_likelihood: float


@dataclass
class CalibrationResult:
    """Output of joint item calibration."""

    items: List[IRTItem]
    thetas: Dict[str, float]
    log_likelihood: float
    epochs_run: int
    converged: bool


def prob_correct(theta: float, discrimination: float, difficulty: float) -> float:
    """2PL response probability."""
    z = discrimination * (theta - difficulty)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def fisher_information(theta: float, item: IRTItem) -> float:
    """Fisher information of ``item`` at ability ``theta``: a^2 * P * Q."""
    p = prob_correct(theta, item.discrimination, item.difficulty)
    return item.discrimination ** 2 * p * (1.0 - p)


class AbilityEstimator:
    """Maximum-likelihood theta estimation via Newton–Raphson."""

    def __init__(
        self,
        theta_min: float = IRT_THETA_MIN,
        theta_max: float = IRT_THETA_MAX,
        max_iter: int = IRT_NEWTON_MAX_ITER,
        tol: float = IRT_NEWTON_TOL,
    ) -> None:
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.max_iter = max_iter
        self.tol = tol

    def estimate(
        self,
        items: Sequence[IRTItem],
        responses: Sequence[bool],
        initial_theta: float = 0.0,
        use_prior: bool = True,
    ) -> AbilityEstimate:
        """Estimate ability from parallel ``items`` / ``responses`` lists.

        Newton–Raphson on the log-likelihood; a weak N(0,1) prior term is
        included (MAP) so extreme response patterns (all correct / all
        incorrect) still yield finite estimates.
        """
        if len(items) != len(responses):
            raise ValueError("items and responses must have equal length")
        if not items:
            return AbilityEstimate(0.0, 1.0, 0, 0.0)

        theta = initial_theta
        for _ in range(self.max_iter):
            grad = 0.0
            hess = 0.0
            for item, resp in zip(items, responses):
                p = prob_correct(theta, item.discrimination, item.difficulty)
                y = 1.0 if resp else 0.0
                grad += item.discrimination * (y - p)
                hess -= item.discrimination ** 2 * p * (1.0 - p)
            if use_prior:
                grad += -theta            # derivative of -theta^2 / 2
                hess += -1.0
            if hess == 0:
                break
            step = grad / hess
            theta_new = theta - step
            theta_new = max(self.theta_min, min(self.theta_max, theta_new))
            if abs(theta_new - theta) < self.tol:
                theta = theta_new
                break
            theta = theta_new

        # Standard error from total Fisher information at the estimate.
        info = sum(fisher_information(theta, it) for it in items)
        if use_prior:
            info += 1.0
        se = 1.0 / math.sqrt(max(info, _EPS))
        se = max(IRT_THETA_SE_FLOOR, se)

        ll = 0.0
        for item, resp in zip(items, responses):
            p = min(max(prob_correct(theta, item.discrimination, item.difficulty), _EPS), 1 - _EPS)
            ll += math.log(p) if resp else math.log(1 - p)
        return AbilityEstimate(
            theta=round(theta, 4),
            standard_error=round(se, 4),
            n_items=len(items),
            log_likelihood=round(ll, 4),
        )


class ItemCalibrator:
    """Joint ML calibration of item parameters from a response matrix."""

    def __init__(
        self,
        learning_rate: float = IRT_CALIBRATION_LR,
        l2: float = IRT_CALIBRATION_L2,
        epochs: int = IRT_CALIBRATION_EPOCHS,
    ) -> None:
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self._estimator = AbilityEstimator()

    def calibrate(
        self,
        student_ids: Sequence[str],
        items: Sequence[IRTItem],
        response_matrix: Sequence[Sequence[Optional[bool]]],
    ) -> CalibrationResult:
        """Calibrate ``items`` from ``response_matrix[student][item]``.

        ``None`` entries denote un-administered items.  Alternating
        optimisation: (1) gradient-ascent step on every item's ``(a, b)``
        given current thetas, (2) Newton re-estimation of every theta.
        Identifiability is anchored by fixing the theta distribution to
        mean 0 / sd 1 after each epoch.
        """
        if len(student_ids) != len(response_matrix):
            raise ValueError("student_ids and response_matrix row count mismatch")
        for row in response_matrix:
            if len(row) != len(items):
                raise ValueError("each response row must match the item count")

        items = [IRTItem(i.item_id, i.concept_id, i.discrimination, i.difficulty) for i in items]
        thetas: Dict[str, float] = {sid: 0.0 for sid in student_ids}
        prev_ll = -math.inf
        converged = False
        epochs_run = 0

        for epoch in range(self.epochs):
            epochs_run = epoch + 1
            # --- item gradient step -------------------------------------
            for j, item in enumerate(items):
                grad_a = 0.0
                grad_b = 0.0
                for si, sid in enumerate(student_ids):
                    resp = response_matrix[si][j]
                    if resp is None:
                        continue
                    theta = thetas[sid]
                    p = prob_correct(theta, item.discrimination, item.difficulty)
                    y = 1.0 if resp else 0.0
                    grad_a += (y - p) * (theta - item.difficulty)
                    grad_b += (y - p) * (-item.discrimination)
                grad_a -= self.l2 * (item.discrimination - 1.0)
                item.discrimination = max(
                    0.1, min(3.0, item.discrimination + self.learning_rate * grad_a)
                )
                item.difficulty = max(
                    -4.0, min(4.0, item.difficulty + self.learning_rate * grad_b)
                )
            # --- theta re-estimation ------------------------------------
            for si, sid in enumerate(student_ids):
                administered = [
                    (items[j], response_matrix[si][j])
                    for j in range(len(items))
                    if response_matrix[si][j] is not None
                ]
                if administered:
                    est = self._estimator.estimate(
                        [a for a, _ in administered],
                        [bool(r) for _, r in administered],
                        initial_theta=thetas[sid],
                    )
                    thetas[sid] = est.theta
            # --- anchor identifiability: standardise thetas -------------
            vals = list(thetas.values())
            if len(vals) > 1:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                sd = math.sqrt(var) or 1.0
                for sid in thetas:
                    thetas[sid] = (thetas[sid] - mean) / sd
            # --- convergence check --------------------------------------
            ll = self._total_log_likelihood(items, thetas, student_ids, response_matrix)
            if abs(ll - prev_ll) < 1e-4:
                converged = True
                prev_ll = ll
                break
            prev_ll = ll

        logger.info(
            "IRT calibration done: epochs=%d converged=%s LL=%.3f",
            epochs_run, converged, prev_ll,
        )
        return CalibrationResult(
            items=items,
            thetas={k: round(v, 4) for k, v in thetas.items()},
            log_likelihood=round(prev_ll, 4),
            epochs_run=epochs_run,
            converged=converged,
        )

    @staticmethod
    def _total_log_likelihood(
        items: Sequence[IRTItem],
        thetas: Dict[str, float],
        student_ids: Sequence[str],
        response_matrix: Sequence[Sequence[Optional[bool]]],
    ) -> float:
        ll = 0.0
        for si, sid in enumerate(student_ids):
            for j, item in enumerate(items):
                resp = response_matrix[si][j]
                if resp is None:
                    continue
                p = min(max(prob_correct(thetas[sid], item.discrimination, item.difficulty), _EPS), 1 - _EPS)
                ll += math.log(p) if resp else math.log(1 - p)
        return ll


class AdaptiveItemSelector:
    """Computerised-adaptive-testing style next-item selection."""

    def __init__(self, estimator: AbilityEstimator | None = None) -> None:
        self.estimator = estimator or AbilityEstimator()

    def select_next(
        self,
        theta: float,
        candidate_items: Sequence[IRTItem],
        used_item_ids: set[str] | None = None,
    ) -> Optional[IRTItem]:
        """Return the unused item with maximal Fisher information at ``theta``."""
        used = used_item_ids or set()
        best: Optional[IRTItem] = None
        best_info = -1.0
        for item in candidate_items:
            if item.item_id in used:
                continue
            info = fisher_information(theta, item)
            if info > best_info:
                best_info = info
                best = item
        return best

    def run_adaptive_session(
        self,
        item_bank: Sequence[IRTItem],
        respond: "object",
        max_items: int = 10,
        target_se: float = 0.30,
    ) -> Tuple[AbilityEstimate, List[str]]:
        """Simulate an adaptive session.

        ``respond`` is a callable ``(IRTItem) -> bool`` (e.g. a real student
        endpoint or a simulator).  Stops early when the standard error of
        theta drops below ``target_se``.  Returns the final estimate and
        the administered item ids in order.
        """
        used: set[str] = set()
        administered_items: List[IRTItem] = []
        administered_responses: List[bool] = []
        estimate = AbilityEstimate(0.0, 1.0, 0, 0.0)
        theta = 0.0
        for _ in range(max_items):
            nxt = self.select_next(theta, item_bank, used)
            if nxt is None:
                break
            answer = bool(respond(nxt))  # type: ignore[operator]
            used.add(nxt.item_id)
            administered_items.append(nxt)
            administered_responses.append(answer)
            estimate = self.estimator.estimate(
                administered_items, administered_responses, initial_theta=theta
            )
            theta = estimate.theta
            if estimate.standard_error <= target_se:
                break
        return estimate, [i.item_id for i in administered_items]
