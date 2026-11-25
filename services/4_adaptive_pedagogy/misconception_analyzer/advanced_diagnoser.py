"""
Advanced misconception diagnoser — confidence-scored diagnoses.
================================================================
Extends the v1 :class:`ErrorDiagnoser` (which it subclasses, so every v1
method keeps working) with:

* **Confidence-scored diagnoses** — each recorded misconception is scored
  by recurrence: a single slip is weak evidence (confidence ~0.45), while
  repeated occurrences across items — especially across *different*
  concepts — raise confidence toward certainty and mark the pattern as
  *persistent*.
* **Bloom-mapped remediation recommendations** — every diagnosis is
  joined with the pattern library to produce a remediation action pitched
  at the Bloom level where the underlying cognitive gap lives.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from common.contracts import BloomLevel

from misconception_analyzer.error_diagnoser import ErrorDiagnoser
from misconception_analyzer.pattern_library import PATTERN_LIBRARY, get_pattern
from config import (
    MISCONCEPTION_CONFIDENCE_BASE,
    MISCONCEPTION_CONFIDENCE_PER_REPEAT,
    MISCONCEPTION_PERSISTENT_COUNT,
)

logger = logging.getLogger(__name__)


class DiagnosedMisconception(BaseModel):
    """A misconception with an evidence-based confidence score."""

    misconception_id: str
    pattern_name_ar: Optional[str] = None
    category: Optional[str] = None
    concept_id: str
    occurrences: int
    confidence: float               # 0..1
    is_persistent: bool
    severity: float                 # from the pattern library (0..1)
    risk_score: float               # confidence * severity
    remediation_bloom_level: Optional[str] = None
    remediation_action_ar: Optional[str] = None
    related_patterns: List[str] = Field(default_factory=list)


class AdvancedErrorDiagnoser(ErrorDiagnoser):
    """ErrorDiagnoser + confidence scoring and remediation mapping."""

    def confidence_for(self, occurrences: int, concept_spread: int = 1) -> float:
        """Confidence grows with recurrence and cross-concept spread.

        ``confidence = base + per_repeat * (occurrences - 1) + 0.07 * (spread - 1)``
        capped at 0.98.
        """
        if occurrences <= 0:
            return 0.0
        conf = (
            MISCONCEPTION_CONFIDENCE_BASE
            + MISCONCEPTION_CONFIDENCE_PER_REPEAT * (occurrences - 1)
            + 0.07 * max(0, concept_spread - 1)
        )
        return float(min(0.98, conf))

    def diagnose_student(
        self, student_id: str, min_confidence: float = 0.0
    ) -> List[DiagnosedMisconception]:
        """All diagnoses for a student, highest risk first."""
        patterns = self.get_error_patterns(student_id)
        # Cross-concept spread per misconception id.
        spread: Dict[str, int] = {}
        for concept_id, ids in patterns.items():
            for mid in set(ids):
                spread[mid] = spread.get(mid, 0) + 1

        diagnoses: List[DiagnosedMisconception] = []
        for concept_id, ids in patterns.items():
            counts: Dict[str, int] = {}
            for mid in ids:
                counts[mid] = counts.get(mid, 0) + 1
            for mid, count in counts.items():
                pattern = get_pattern(mid)
                confidence = self.confidence_for(count, spread.get(mid, 1))
                if confidence < min_confidence:
                    continue
                severity = pattern.severity if pattern else 0.5
                diagnoses.append(
                    DiagnosedMisconception(
                        misconception_id=mid,
                        pattern_name_ar=pattern.name_ar if pattern else None,
                        category=pattern.category if pattern else None,
                        concept_id=concept_id,
                        occurrences=count,
                        confidence=round(confidence, 4),
                        is_persistent=count >= MISCONCEPTION_PERSISTENT_COUNT,
                        severity=severity,
                        risk_score=round(confidence * severity, 4),
                        remediation_bloom_level=(
                            pattern.remediation_bloom_level if pattern else None
                        ),
                        remediation_action_ar=(
                            pattern.remediation_action_ar if pattern else None
                        ),
                        related_patterns=pattern.related_patterns if pattern else [],
                    )
                )
        diagnoses.sort(key=lambda d: d.risk_score, reverse=True)
        return diagnoses

    def persistent_misconceptions(
        self, student_id: str
    ) -> List[DiagnosedMisconception]:
        """Only the persistent (recurring) diagnoses for a student."""
        return [d for d in self.diagnose_student(student_id) if d.is_persistent]

    def remediation_recommendations(
        self, student_id: str, top_n: int = 5
    ) -> List[Dict[str, object]]:
        """Top remediation actions, grouped by the Bloom level to pitch at."""
        recommendations: List[Dict[str, object]] = []
        for diag in self.diagnose_student(student_id)[:top_n]:
            level = diag.remediation_bloom_level or BloomLevel.UNDERSTAND.value
            recommendations.append(
                {
                    "misconception_id": diag.misconception_id,
                    "concept_id": diag.concept_id,
                    "confidence": diag.confidence,
                    "risk_score": diag.risk_score,
                    "target_bloom_level": level,
                    "action_ar": diag.remediation_action_ar
                    or "مراجعة موجّهة للمفهوم مع أسئلة تشخيصية إضافية.",
                }
            )
        return recommendations

    @staticmethod
    def known_pattern_ids() -> List[str]:
        """All misconception ids known to the pattern library."""
        return sorted(PATTERN_LIBRARY.keys())
