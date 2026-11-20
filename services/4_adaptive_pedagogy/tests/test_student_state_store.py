"""
Tests for Student State Store.
"""
import sys
from pathlib import Path

_service_root = Path(__file__).resolve().parent.parent
if str(_service_root) not in sys.path:
    sys.path.insert(0, str(_service_root))

import os
import tempfile
from common.contracts import StudentConceptMastery
from bkt_engine.student_state_store import StudentStateStore


def test_get_or_create_state():
    store = StudentStateStore()
    state = store.get_or_create_state("std_1", "concept_bge", base_difficulty=0.7)
    assert state.student_id == "std_1"
    assert state.concept_id == "concept_bge"
    assert state.bkt_p_mastery > 0.0
    assert state.fsrs_difficulty == 7.0

    state2 = store.get_or_create_state("std_1", "concept_bge")
    assert state2.bkt_p_mastery == state.bkt_p_mastery


def test_update_and_summary():
    store = StudentStateStore()
    s1 = store.get_or_create_state("std_1", "c1")
    s2 = store.get_or_create_state("std_1", "c2")

    store.update_state(s1.model_copy(update={"bkt_p_mastery": 0.98}))
    store.update_state(s2.model_copy(update={"bkt_p_mastery": 0.75}))

    summary = store.get_mastery_summary("std_1")
    assert summary["total_concepts"] == 2
    assert summary["mastered_count"] == 1
    assert summary["learning_count"] == 1
    assert summary["unlearned_count"] == 0

    mastered = store.get_mastered_concepts("std_1")
    assert len(mastered) == 1
    assert mastered[0].concept_id == "c1"


def test_json_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_states.json")
        store1 = StudentStateStore(persistence_filepath=filepath)
        s1 = store1.get_or_create_state("std_persist", "c_persist")
        store1.update_state(s1.model_copy(update={"bkt_p_mastery": 0.88}))

        store2 = StudentStateStore(persistence_filepath=filepath)
        retrieved = store2.get_state("std_persist", "c_persist")
        assert retrieved is not None
        assert retrieved.bkt_p_mastery == 0.88
