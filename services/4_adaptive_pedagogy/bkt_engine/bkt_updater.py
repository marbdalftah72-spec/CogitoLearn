"""
Bayesian Knowledge Tracing (BKT) Engine for CogitoLearn.
========================================================
This module implements the standard 4-parameter BKT probabilistic model:
- P(L0): Initial probability of concept mastery
- P(T): Transition probability (learning rate from unlearned to learned)
- P(G): Guess probability (correct answer despite not knowing concept)
- P(S): Slip probability (incorrect answer despite knowing concept)

Mathematical Formulations:
1. Posterior Update upon Observation:
   P(L_t | Correct)   = P(L_t) * (1 - P(S)) / [ P(L_t) * (1 - P(S)) + (1 - P(L_t)) * P(G) ]
   P(L_t | Incorrect) = P(L_t) * P(S)       / [ P(L_t) * P(S)       + (1 - P(L_t)) * (1 - P(G)) ]

2. Latent State Transition:
   P(L_{t+1}) = P(L_t | Obs) + (1 - P(L_t | Obs)) * P(T)
"""

from typing import List, Literal
from common.contracts import StudentConceptMastery
from common.exceptions.domain_errors import PedagogyStateError

MasteryClass = Literal["MASTERED", "LEARNING", "UNLEARNED"]


class BKTUpdater:
    """
    Standard Bayesian Knowledge Tracing updater.
    Operates immutably on StudentConceptMastery contracts.
    """

    def __init__(
        self,
        default_p_l0: float = 0.15,
        default_p_transit: float = 0.15,
        default_p_guess: float = 0.20,
        default_p_slip: float = 0.10,
        mastery_threshold: float = 0.95,
        learning_threshold: float = 0.50,
    ):
        self.default_p_l0 = default_p_l0
        self.default_p_transit = default_p_transit
        self.default_p_guess = default_p_guess
        self.default_p_slip = default_p_slip
        self.mastery_threshold = mastery_threshold
        self.learning_threshold = learning_threshold

    def predict_correct_probability(self, state: StudentConceptMastery) -> float:
        """
        Calculates P(Correct) = P(L) * (1 - P(S)) + (1 - P(L)) * P(G)
        """
        p_l = state.bkt_p_mastery
        p_s = state.bkt_p_slip
        p_g = state.bkt_p_guess

        p_correct = (p_l * (1.0 - p_s)) + ((1.0 - p_l) * p_g)
        return float(max(0.001, min(0.999, p_correct)))

    def update_mastery(
        self, state: StudentConceptMastery, is_correct: bool
    ) -> StudentConceptMastery:
        """
        Updates student mastery given a binary outcome (is_correct).
        Returns a new StudentConceptMastery instance with updated p_mastery.
        """
        p_l = float(state.bkt_p_mastery)
        p_s = float(state.bkt_p_slip)
        p_g = float(state.bkt_p_guess)
        p_t = float(state.bkt_p_transit)

        # Validate probabilities
        for name, val in [("p_mastery", p_l), ("p_slip", p_s), ("p_guess", p_g), ("p_transit", p_t)]:
            if not (0.0 <= val <= 1.0):
                raise PedagogyStateError(f"Invalid probability for {name}: {val}. Must be in [0, 1].")

        # Step 1: Posterior update given evidence
        if is_correct:
            numerator = p_l * (1.0 - p_s)
            denominator = numerator + ((1.0 - p_l) * p_g)
        else:
            numerator = p_l * p_s
            denominator = numerator + ((1.0 - p_l) * (1.0 - p_g))

        if denominator <= 0.0:
            p_posterior = p_l
        else:
            p_posterior = numerator / denominator

        # Step 2: Learning transition update
        p_next = p_posterior + ((1.0 - p_posterior) * p_t)

        # Clamp to strictly valid probability boundaries
        p_next = float(max(0.0001, min(0.9999, p_next)))

        # Return updated state
        updated = state.model_copy(
            update={
                "bkt_p_mastery": round(p_next, 4),
            }
        )
        return updated

    def batch_update(
        self, state: StudentConceptMastery, responses: List[bool]
    ) -> StudentConceptMastery:
        """
        Sequentially updates mastery for a sequence of binary responses.
        """
        current_state = state
        for outcome in responses:
            current_state = self.update_mastery(current_state, outcome)
        return current_state

    def classify_mastery(self, p_mastery: float) -> MasteryClass:
        """
        Categorizes mastery level:
        - MASTERED: p_mastery >= 0.95
        - LEARNING: 0.50 <= p_mastery < 0.95
        - UNLEARNED: p_mastery < 0.50
        """
        if p_mastery >= self.mastery_threshold:
            return "MASTERED"
        elif p_mastery >= self.learning_threshold:
            return "LEARNING"
        else:
            return "UNLEARNED"
