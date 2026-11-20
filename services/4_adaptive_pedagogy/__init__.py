"""
CogitoLearn Service 4: Adaptive Pedagogy & Assessment Intelligence.
"""
import sys
from pathlib import Path

_service_root = Path(__file__).resolve().parent
if str(_service_root) not in sys.path:
    sys.path.insert(0, str(_service_root))

from pedagogy_coordinator import (
    PedagogyCoordinator,
    AssessmentPackage,
    ResponseFeedback,
    EssayFeedback,
    StudentDashboard,
)
from bkt_engine.bkt_updater import BKTUpdater
from bkt_engine.student_state_store import StudentStateStore
from fsrs_scheduler.memory_optimizer import FSRSScheduler
from misconception_analyzer.error_diagnoser import (
    ErrorDiagnoser,
    RemediationPlan,
    RemediationStep,
)
from bloom_engine.mcq_generator import MCQGenerator
from bloom_engine.essay_evaluator import (
    EssayEvaluator,
    EssayEvaluationResult,
)

__all__ = [
    "PedagogyCoordinator",
    "AssessmentPackage",
    "ResponseFeedback",
    "EssayFeedback",
    "StudentDashboard",
    "BKTUpdater",
    "StudentStateStore",
    "FSRSScheduler",
    "ErrorDiagnoser",
    "RemediationPlan",
    "RemediationStep",
    "MCQGenerator",
    "EssayEvaluator",
    "EssayEvaluationResult",
]
