"""
FSRS-4.5 (Free Spaced Repetition Scheduler) Memory Optimizer.
=============================================================
Computes memory retrievability, updates stability and difficulty, and
calculates optimal spaced repetition intervals calibrated for university courses.

Ratings:
1 = Again (Failed recall / Lapse)
2 = Hard  (Recalled with severe effort)
3 = Good  (Successful recall with appropriate effort)
4 = Easy  (Effortless recall)
"""

import math
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from common.contracts import StudentConceptMastery


class FSRSScheduler:
    """
    Implements FSRS-4.5 spaced repetition scheduling.
    """

    def __init__(self, desired_retention: float = 0.90):
        self.desired_retention = desired_retention

    def calculate_retrievability(self, stability: float, days_elapsed: float) -> float:
        """
        Calculates probability of recall R(t) = (1 + t / (9 * S))^(-1)
        """
        if stability <= 0.0:
            return 0.0
        if days_elapsed <= 0.0:
            return 1.0

        r = (1.0 + (days_elapsed / (9.0 * stability))) ** (-1.0)
        return float(max(0.0, min(1.0, r)))

    def get_optimal_interval(
        self, stability: float, desired_retention: Optional[float] = None
    ) -> float:
        """
        Calculates optimal review interval in days for a target retention rate.
        Interval = 9 * S * (1 / R - 1)
        """
        target_r = desired_retention or self.desired_retention
        if target_r <= 0.0 or target_r >= 1.0:
            target_r = 0.90

        interval = 9.0 * stability * ((1.0 / target_r) - 1.0)
        return float(max(1.0, interval))

    def schedule_review(
        self,
        state: StudentConceptMastery,
        rating: int,
        review_date: Optional[datetime] = None,
    ) -> StudentConceptMastery:
        """
        Processes a review session and produces updated stability, difficulty,
        and next_review_due timestamp.
        """
        if rating not in [1, 2, 3, 4]:
            raise ValueError(f"Invalid rating {rating}. Must be 1 (Again), 2 (Hard), 3 (Good), or 4 (Easy).")

        now = review_date or datetime.now(timezone.utc)
        curr_stability = max(0.1, state.fsrs_stability)
        curr_difficulty = max(1.0, min(10.0, state.fsrs_difficulty))

        # Calculate days elapsed since previous review if recorded
        days_elapsed = 1.0
        if state.next_review_due:
            try:
                prev_due = datetime.fromisoformat(state.next_review_due)
                # Approximate days since previous review based on due date
                days_diff = (now - prev_due).total_seconds() / 86400.0
                days_elapsed = max(0.5, curr_stability + days_diff)
            except Exception:
                days_elapsed = curr_stability

        # Current retrievability
        r = self.calculate_retrievability(curr_stability, days_elapsed)

        # Update difficulty D:
        # D' = D + 0.5 * (3 - rating)
        d_delta = 0.5 * (3 - rating)
        new_difficulty = max(1.0, min(10.0, curr_difficulty + d_delta))

        # Update stability S:
        if rating == 1:
            # Lapse / Again: Memory failure
            # Stability contracts sharply to encourage rapid reinforcement
            new_stability = max(0.3, curr_stability * 0.25)
        else:
            # Successful recall (Hard=2, Good=3, Easy=4)
            grade_modifiers = {2: 0.65, 3: 1.0, 4: 1.35}
            modifier = grade_modifiers.get(rating, 1.0)

            # FSRS exponential growth formula
            growth_exponent = (
                math.exp(0.1)
                * (11.0 - new_difficulty)
                * (curr_stability ** (-0.2))
                * (math.exp(0.08 * (1.0 - r)) - 1.0)
                * modifier
            )
            # Ensure stability increases on success
            growth_factor = max(1.15, 1.0 + growth_exponent)
            new_stability = curr_stability * growth_factor

        # Determine next interval
        optimal_interval_days = self.get_optimal_interval(new_stability, self.desired_retention)
        days_to_add = int(round(optimal_interval_days))
        next_due = now + timedelta(days=days_to_add)

        return state.model_copy(
            update={
                "fsrs_stability": round(new_stability, 3),
                "fsrs_difficulty": round(new_difficulty, 2),
                "next_review_due": next_due.isoformat(),
            }
        )

    def get_due_concepts(
        self,
        states: List[StudentConceptMastery],
        current_date: Optional[datetime] = None,
    ) -> List[StudentConceptMastery]:
        """
        Filters concepts that are currently due for review.
        """
        now = current_date or datetime.now(timezone.utc)
        due_list = []
        for s in states:
            if not s.next_review_due:
                # Never reviewed -> Due immediately
                due_list.append(s)
            else:
                try:
                    due_dt = datetime.fromisoformat(s.next_review_due)
                    if due_dt <= now:
                        due_list.append(s)
                except Exception:
                    due_list.append(s)
        return due_list

    def bulk_schedule(
        self,
        states: List[StudentConceptMastery],
        ratings: Dict[str, int],
        current_date: Optional[datetime] = None,
    ) -> List[StudentConceptMastery]:
        """
        Schedules reviews across multiple concepts keyed by concept_id.
        """
        updated_states = []
        for state in states:
            rating = ratings.get(state.concept_id)
            if rating is not None:
                updated = self.schedule_review(state, rating, review_date=current_date)
                updated_states.append(updated)
            else:
                updated_states.append(state)
        return updated_states
