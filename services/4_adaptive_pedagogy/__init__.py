"""
CogitoLearn Service 4: Adaptive Pedagogy & Assessment Intelligence.

v2 (advanced professional edition) adds: fitted BKT with forgetting decay
and prediction evaluation, a 2PL IRT engine with adaptive item selection,
an FSRS-5 scheduler with optimal-retention solving and workload
forecasting, a multi-criteria essay evaluator, classical item analysis,
confidence-scored misconception diagnosis, and prerequisite-graph
learning-path planning — all fused by ``PedagogyCoordinator`` v2 methods.
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
    UnifiedStudentReport,
)
from bkt_engine.bkt_updater import BKTUpdater
from bkt_engine.student_state_store import StudentStateStore
from bkt_engine.bkt_fitter import (
    BKTParameters,
    BKTParameterFitter,
    BKTPredictionEvaluator,
)
from fsrs_scheduler.memory_optimizer import FSRSScheduler
from fsrs_scheduler.fsrs5_optimizer import FSRS5Scheduler
from misconception_analyzer.error_diagnoser import (
    ErrorDiagnoser,
    RemediationPlan,
    RemediationStep,
)
from misconception_analyzer.advanced_diagnoser import (
    AdvancedErrorDiagnoser,
    DiagnosedMisconception,
)
from bloom_engine.mcq_generator import MCQGenerator
from bloom_engine.essay_evaluator import (
    EssayEvaluator,
    EssayEvaluationResult,
)
from bloom_engine.essay_evaluator_v2 import EssayEvaluatorV2
from irt_engine.irt_model import (
    AbilityEstimator,
    AdaptiveItemSelector,
    IRTItem,
    ItemCalibrator,
)
from learning_path.path_planner import LearningPathPlanner, SessionPlan

__all__ = [
    # coordinator
    "PedagogyCoordinator",
    "AssessmentPackage",
    "ResponseFeedback",
    "EssayFeedback",
    "StudentDashboard",
    "UnifiedStudentReport",
    # BKT
    "BKTUpdater",
    "StudentStateStore",
    "BKTParameters",
    "BKTParameterFitter",
    "BKTPredictionEvaluator",
    # FSRS
    "FSRSScheduler",
    "FSRS5Scheduler",
    # misconceptions
    "ErrorDiagnoser",
    "AdvancedErrorDiagnoser",
    "DiagnosedMisconception",
    "RemediationPlan",
    "RemediationStep",
    # Bloom
    "MCQGenerator",
    "EssayEvaluator",
    "EssayEvaluationResult",
    "EssayEvaluatorV2",
    # IRT
    "AbilityEstimator",
    "AdaptiveItemSelector",
    "IRTItem",
    "ItemCalibrator",
    # learning path
    "LearningPathPlanner",
    "SessionPlan",
]
