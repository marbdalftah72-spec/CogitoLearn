"""
Central configuration for CogitoLearn Service 4 (Adaptive Pedagogy) v2.
=======================================================================
Single source of truth for every tunable weight, threshold and algorithmic
constant used by the v2 engines (fitted BKT, 2PL IRT, FSRS-5, Bloom essay
evaluator v2, item analysis, misconception diagnosis and learning-path
planning).

All values are plain module-level constants plus an immutable
:class:`PedagogyConfig` dataclass so callers can override a coherent bundle
of settings without mutating globals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BKT (Bayesian Knowledge Tracing)
# ---------------------------------------------------------------------------
BKT_DEFAULT_P_L0: float = 0.15
BKT_DEFAULT_P_TRANSIT: float = 0.15
BKT_DEFAULT_P_GUESS: float = 0.20
BKT_DEFAULT_P_SLIP: float = 0.10
BKT_MASTERY_THRESHOLD: float = 0.95
BKT_LEARNING_THRESHOLD: float = 0.50
#: Half-life (days) of the exponential forgetting decay applied between
#: study sessions when no review has occurred:  P <- P * 0.5 ** (days / HL)
BKT_FORGETTING_HALF_LIFE_DAYS: float = 21.0
#: Grid resolution used by the BKT parameter grid search.
BKT_FIT_GRID: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)

# ---------------------------------------------------------------------------
# IRT (Item Response Theory, 2PL)
# ---------------------------------------------------------------------------
IRT_THETA_MIN: float = -4.0
IRT_THETA_MAX: float = 4.0
IRT_THETA_SE_FLOOR: float = 0.05
IRT_NEWTON_MAX_ITER: int = 60
IRT_NEWTON_TOL: float = 1e-5
#: L2 regularisation strength used during joint item calibration.
IRT_CALIBRATION_L2: float = 0.10
IRT_CALIBRATION_LR: float = 0.05
IRT_CALIBRATION_EPOCHS: int = 200

# ---------------------------------------------------------------------------
# FSRS-5
# ---------------------------------------------------------------------------
FSRS5_DEFAULT_RETENTION: float = 0.90
#: The 17 published FSRS-5 default weights (w0..w16).
FSRS5_DEFAULT_WEIGHTS: Tuple[float, ...] = (
    0.40255, 1.18385, 3.173, 15.69105,  # w0..w3  initial stability per grade
    7.1949, 0.5345,                     # w4, w5  difficulty
    1.4604, 0.0046,                     # w6, w7  difficulty update / mean reversion
    1.54575, 0.1192, 1.01925,           # w8..w10 stability after success
    1.9395, 0.11, 0.29605, 2.2698,      # w11..w14 stability after lapse
    0.2315, 2.9898,                     # w15, w16 hard / easy bonuses
)
FSRS5_DECAY: float = -0.5
FSRS5_FACTOR: float = 19.0 / 81.0
FSRS_MIN_STABILITY: float = 0.1
FSRS_MAX_INTERVAL_DAYS: float = 365.0

# ---------------------------------------------------------------------------
# Bloom essay evaluator v2
# ---------------------------------------------------------------------------
ESSAY_V2_WEIGHTS: Dict[str, float] = {
    "content_coverage": 0.40,
    "structure": 0.20,
    "bloom_verbs": 0.25,
    "coherence": 0.15,
}
ESSAY_MIN_WORDS_FOR_FULL_STRUCTURE: int = 120

# ---------------------------------------------------------------------------
# Item analysis (classical test theory)
# ---------------------------------------------------------------------------
ITEM_DIFFICULTY_IDEAL_MIN: float = 0.30
ITEM_DIFFICULTY_IDEAL_MAX: float = 0.80
ITEM_DISCRIMINATION_GOOD: float = 0.30
ITEM_DISCRIMINATION_POOR: float = 0.10

# ---------------------------------------------------------------------------
# Misconception diagnosis
# ---------------------------------------------------------------------------
MISCONCEPTION_CONFIDENCE_BASE: float = 0.45
MISCONCEPTION_CONFIDENCE_PER_REPEAT: float = 0.18
MISCONCEPTION_PERSISTENT_COUNT: int = 3

# ---------------------------------------------------------------------------
# Learning-path planning
# ---------------------------------------------------------------------------
PATH_MASTERY_GATE: float = 0.65
PATH_SESSION_MAX_NEW: int = 3
PATH_SESSION_MAX_REVIEW: int = 5


@dataclass(frozen=True)
class PedagogyConfig:
    """Immutable bundle of Service-4 v2 tunables.

    Engines accept an optional ``PedagogyConfig``; when omitted they fall
    back to the module-level defaults above, so v1 behaviour is preserved.
    """

    bkt_mastery_threshold: float = BKT_MASTERY_THRESHOLD
    bkt_learning_threshold: float = BKT_LEARNING_THRESHOLD
    bkt_forgetting_half_life_days: float = BKT_FORGETTING_HALF_LIFE_DAYS
    irt_theta_bounds: Tuple[float, float] = (IRT_THETA_MIN, IRT_THETA_MAX)
    fsrs5_weights: Tuple[float, ...] = FSRS5_DEFAULT_WEIGHTS
    fsrs5_default_retention: float = FSRS5_DEFAULT_RETENTION
    essay_v2_weights: Dict[str, float] = field(
        default_factory=lambda: dict(ESSAY_V2_WEIGHTS)
    )
    path_mastery_gate: float = PATH_MASTERY_GATE
    path_session_max_new: int = PATH_SESSION_MAX_NEW
    path_session_max_review: int = PATH_SESSION_MAX_REVIEW

    def validate(self) -> "PedagogyConfig":
        """Validate the bundle, raising ``ValueError`` on inconsistent values."""
        total = sum(self.essay_v2_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"essay_v2_weights must sum to 1.0, got {total:.4f}")
        if not (0.0 < self.bkt_learning_threshold < self.bkt_mastery_threshold <= 1.0):
            raise ValueError("BKT thresholds must satisfy 0 < learning < mastery <= 1")
        if len(self.fsrs5_weights) != 17:
            raise ValueError("FSRS-5 requires exactly 17 weights (w0..w16)")
        logger.debug("PedagogyConfig validated: %s", self)
        return self


#: Shared default configuration instance used across Service 4 v2.
DEFAULT_CONFIG = PedagogyConfig()
