"""
Core Contract & Schema Integrity Tests
======================================
Ensures that all domain models can be instantiated and validated properly.
"""

import pytest
from common.contracts import (
    MediaModality,
    EvidenceSpan,
    UnifiedDocumentAST,
    BloomLevel,
    MCQOption,
    MCQQuestion,
    StudentConceptMastery
)

def test_evidence_span_creation():
    span = EvidenceSpan(
        span_id="span-test-001",
        lecture_id="lec-001",
        hierarchy_level=0,
        content_text="Self-attention mechanisms calculate relationships between all words in parallel."
    )
    assert span.span_id == "span-test-001"
    assert span.hierarchy_level == 0

def test_mcq_distractor_mapping():
    option_wrong = MCQOption(
        option_id="opt-01",
        text="Reduces vocabulary size",
        is_correct=False,
        misconception_id="MISC_TOKENIZATION_CONFUSION",
        misconception_explanation="Confusing positional encoding with tokenizer vocabulary reduction."
    )
    assert option_wrong.is_correct is False
    assert option_wrong.misconception_id == "MISC_TOKENIZATION_CONFUSION"

def test_student_mastery_initial_state():
    student_state = StudentConceptMastery(
        student_id="student-101",
        concept_id="CONCEPT_ATTENTION_TRANSFORMERS"
    )
    assert student_state.bkt_p_mastery == 0.15
    assert student_state.fsrs_stability == 1.0
