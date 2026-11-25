"""
FSRS-5 scheduler (Free Spaced Repetition Scheduler, version 5).
================================================================
Upgrades the v1 FSRS-4.5 approximation in ``memory_optimizer.py`` to the
FSRS-5 model:

* Retrievability:  ``R(t, S) = (1 + (19/81) * t / S) ** -0.5``
* Initial stability per first grade: ``S0(G) = w[G-1]``
* Difficulty: ``D0(G) = w4 - exp(w5 * (G - 1)) + 1`` with linear-damped
  updates and mean reversion toward ``D0(Easy)``.
* Post-recall stability growth and post-lapse stability follow the
  published FSRS-5 recurrences with the 17 default weights ``w0..w16``.

Extras beyond the core recurrences:

* **Optimal-retention solver** — grid-searches the retention target that
  minimises simulated total review workload over a horizon.
* **Workload forecaster** — day-by-day projection of due review counts
  for a population of cards, simulating stability growth at each review.
"""

from __future__ import annotations

import logging
import math
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from common.contracts import StudentConceptMastery

from config import (
    FSRS5_DECAY,
    FSRS5_DEFAULT_RETENTION,
    FSRS5_DEFAULT_WEIGHTS,
    FSRS5_FACTOR,
    FSRS_MAX_INTERVAL_DAYS,
    FSRS_MIN_STABILITY,
)

logger = logging.getLogger(__name__)


def _clamp_difficulty(d: float) -> float:
    return float(max(1.0, min(10.0, d)))


class FSRS5Scheduler:
    """FSRS-5 spaced-repetition scheduler operating on mastery contracts."""

    def __init__(
        self,
        weights: Sequence[float] = FSRS5_DEFAULT_WEIGHTS,
        desired_retention: float = FSRS5_DEFAULT_RETENTION,
    ) -> None:
        if len(weights) != 17:
            raise ValueError("FSRS-5 requires exactly 17 weights (w0..w16)")
        self.w: Tuple[float, ...] = tuple(float(x) for x in weights)
        self.desired_retention = desired_retention

    # -- core curves --------------------------------------------------------
    def retrievability(self, days_elapsed: float, stability: float) -> float:
        """R(t, S) = (1 + FACTOR * t / S) ** DECAY."""
        if stability <= 0:
            return 0.0
        if days_elapsed <= 0:
            return 1.0
        r = (1.0 + FSRS5_FACTOR * days_elapsed / stability) ** FSRS5_DECAY
        return float(max(0.0, min(1.0, r)))

    def interval_for_retention(self, stability: float, retention: float) -> float:
        """Invert the forgetting curve: days until R drops to ``retention``."""
        if not (0.0 < retention < 1.0):
            retention = self.desired_retention
        days = (stability / FSRS5_FACTOR) * (retention ** (1.0 / FSRS5_DECAY) - 1.0)
        return float(max(1.0, min(FSRS_MAX_INTERVAL_DAYS, days)))

    # -- difficulty / stability recurrences ---------------------------------
    def initial_difficulty(self, grade: int) -> float:
        """D0(G) = w4 - exp(w5 * (G - 1)) + 1, clamped to [1, 10]."""
        return _clamp_difficulty(self.w[4] - math.exp(self.w[5] * (grade - 1)) + 1.0)

    def initial_stability(self, grade: int) -> float:
        """S0(G) = w[G-1]."""
        return float(max(FSRS_MIN_STABILITY, self.w[grade - 1]))

    def update_difficulty(self, difficulty: float, grade: int) -> float:
        """Linear-damped difficulty update with mean reversion to D0(4)."""
        delta = -self.w[6] * (grade - 3)
        damped = difficulty + delta * (10.0 - difficulty) / 9.0
        anchor = self.initial_difficulty(4)
        reverted = self.w[7] * anchor + (1.0 - self.w[7]) * damped
        return _clamp_difficulty(reverted)

    def stability_after_success(
        self, stability: float, difficulty: float, retrievability: float, grade: int
    ) -> float:
        """FSRS-5 post-recall stability for grades 2..4."""
        hard_factor = self.w[15] if grade == 2 else 1.0
        easy_factor = self.w[16] if grade == 4 else 1.0
        growth = (
            math.exp(self.w[8])
            * (11.0 - difficulty)
            * (stability ** (-self.w[9]))
            * (math.exp(self.w[10] * (1.0 - retrievability)) - 1.0)
            * hard_factor
            * easy_factor
        )
        return float(max(FSRS_MIN_STABILITY, stability * (1.0 + growth)))

    def stability_after_lapse(
        self, stability: float, difficulty: float, retrievability: float
    ) -> float:
        """FSRS-5 post-lapse stability for grade 1 (Again)."""
        s_fail = (
            self.w[11]
            * (difficulty ** (-self.w[12]))
            * ((stability + 1.0) ** self.w[13] - 1.0)
            * math.exp(self.w[14] * (1.0 - retrievability))
        )
        return float(max(FSRS_MIN_STABILITY, min(s_fail, stability)))

    # -- state-level API ------------------------------------------------------
    def review_card(
        self,
        state: StudentConceptMastery,
        rating: int,
        days_elapsed: float,
        review_date: Optional[datetime] = None,
        is_first_review: bool = False,
    ) -> StudentConceptMastery:
        """Apply one FSRS-5 review to ``state`` and return the updated copy."""
        if rating not in (1, 2, 3, 4):
            raise ValueError(f"Invalid rating {rating}; must be 1..4")
        now = review_date or datetime.now(timezone.utc)
        if is_first_review:
            new_stability = self.initial_stability(rating)
            new_difficulty = self.initial_difficulty(rating)
        else:
            stability = max(FSRS_MIN_STABILITY, state.fsrs_stability)
            difficulty = _clamp_difficulty(state.fsrs_difficulty)
            r = self.retrievability(days_elapsed, stability)
            new_difficulty = self.update_difficulty(difficulty, rating)
            if rating == 1:
                new_stability = self.stability_after_lapse(stability, difficulty, r)
            else:
                new_stability = self.stability_after_success(
                    stability, difficulty, r, rating
                )
        interval = self.interval_for_retention(new_stability, self.desired_retention)
        next_due = now + timedelta(days=int(round(interval)))
        return state.model_copy(
            update={
                "fsrs_stability": round(new_stability, 3),
                "fsrs_difficulty": round(new_difficulty, 2),
                "next_review_due": next_due.isoformat(),
            }
        )

    # -- workload simulation --------------------------------------------------
    def _simulate_workload(
        self,
        stabilities: Sequence[float],
        difficulties: Sequence[float],
        retention: float,
        horizon_days: int,
        rng: random.Random,
    ) -> int:
        """Simulate reviews for a card population; return total review count."""
        # next-due day offsets and mutable stability/difficulty per card
        due = [self.interval_for_retention(s, retention) for s in stabilities]
        last = [0.0] * len(stabilities)
        stab = list(stabilities)
        diff = list(difficulties)
        total = 0
        for day in range(1, horizon_days + 1):
            for i in range(len(stab)):
                if due[i] <= day:
                    total += 1
                    elapsed = day - last[i]
                    last[i] = float(day)
                    r = self.retrievability(elapsed, stab[i])
                    if rng.random() < r:
                        grade = 3  # Good
                        stab[i] = self.stability_after_success(stab[i], diff[i], r, grade)
                    else:
                        grade = 1  # Again
                        stab[i] = self.stability_after_lapse(stab[i], diff[i], r)
                    diff[i] = self.update_difficulty(diff[i], grade)
                    due[i] = day + self.interval_for_retention(stab[i], retention)
        return total

    def solve_optimal_retention(
        self,
        stabilities: Sequence[float],
        difficulties: Optional[Sequence[float]] = None,
        horizon_days: int = 180,
        candidates: Optional[Sequence[float]] = None,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Grid-search the retention minimising simulated review workload.

        Higher retention shortens intervals (more reviews but fewer lapses);
        lower retention lengthens them (fewer reviews but more re-learning).
        The workload curve is U-shaped; this returns its minimum.
        """
        diffs = list(difficulties) if difficulties else [5.0] * len(stabilities)
        grid = candidates or [0.70 + 0.02 * k for k in range(15)]  # 0.70..0.98
        results: Dict[float, int] = {}
        for retention in grid:
            rng = random.Random(seed)
            results[retention] = self._simulate_workload(
                stabilities, diffs, retention, horizon_days, rng
            )
        best_retention = min(results, key=lambda k: results[k])
        return {
            "optimal_retention": round(best_retention, 2),
            "expected_reviews": float(results[best_retention]),
            "workload_by_retention": {round(k, 2): float(v) for k, v in results.items()},
        }

    def forecast_workload(
        self,
        states: Sequence[StudentConceptMastery],
        horizon_days: int = 30,
        start_date: Optional[datetime] = None,
    ) -> List[Dict[str, object]]:
        """Project due-review counts per day for the next ``horizon_days``.

        Assumes every due card is reviewed on its due day with a Good (3)
        rating, so stability grows deterministically along the forecast.
        Returns ``[{"date": iso, "due_count": int, "cumulative": int}]``.
        """
        start = (start_date or datetime.now(timezone.utc)).date()
        counts: Dict[int, int] = {d: 0 for d in range(horizon_days)}
        for state in states:
            stability = max(FSRS_MIN_STABILITY, state.fsrs_stability)
            difficulty = _clamp_difficulty(state.fsrs_difficulty)
            # first due offset
            if state.next_review_due:
                try:
                    due_dt = datetime.fromisoformat(state.next_review_due).date()
                    offset = (due_dt - start).days
                except Exception:
                    offset = 0
            else:
                offset = 0
            offset = max(0, offset)
            while offset < horizon_days:
                counts[offset] += 1
                r = self.retrievability(
                    self.interval_for_retention(stability, self.desired_retention),
                    stability,
                )
                stability = self.stability_after_success(stability, difficulty, r, 3)
                difficulty = self.update_difficulty(difficulty, 3)
                offset += int(round(self.interval_for_retention(stability, self.desired_retention)))
        forecast: List[Dict[str, object]] = []
        cumulative = 0
        for d in range(horizon_days):
            cumulative += counts[d]
            forecast.append(
                {
                    "date": (start + timedelta(days=d)).isoformat(),
                    "due_count": counts[d],
                    "cumulative": cumulative,
                }
            )
        return forecast
