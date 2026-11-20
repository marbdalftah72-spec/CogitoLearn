"""
Thread-safe Student State Store for CogitoLearn.
================================================
Manages per-student, per-concept knowledge profiles with in-memory caching
and optional JSON persistence.
"""

import json
import os
import threading
from typing import Dict, List, Optional, Any
from common.contracts import StudentConceptMastery


class StudentStateStore:
    """
    Manages in-memory storage and persistence of StudentConceptMastery records.
    Key format: (student_id, concept_id)
    """

    def __init__(self, persistence_filepath: Optional[str] = None):
        self._lock = threading.Lock()
        self._store: Dict[str, StudentConceptMastery] = {}
        self.persistence_filepath = persistence_filepath

        if self.persistence_filepath and os.path.exists(self.persistence_filepath):
            self.load_from_file(self.persistence_filepath)

    @staticmethod
    def _make_key(student_id: str, concept_id: str) -> str:
        return f"{student_id}::{concept_id}"

    def get_or_create_state(
        self,
        student_id: str,
        concept_id: str,
        base_difficulty: float = 0.50,
    ) -> StudentConceptMastery:
        """
        Retrieves existing state or initializes a new StudentConceptMastery profile.
        """
        key = self._make_key(student_id, concept_id)
        with self._lock:
            if key in self._store:
                return self._store[key]

            # Initialize new state calibrated by base difficulty
            # Harder concepts start with slightly lower p_mastery and higher fsrs_difficulty
            initial_p_mastery = max(0.05, 0.25 - (base_difficulty * 0.15))
            initial_fsrs_diff = max(1.0, min(10.0, base_difficulty * 10.0))

            state = StudentConceptMastery(
                student_id=student_id,
                concept_id=concept_id,
                bkt_p_mastery=round(initial_p_mastery, 4),
                bkt_p_slip=0.10,
                bkt_p_guess=0.20,
                bkt_p_transit=0.15,
                fsrs_stability=1.0,
                fsrs_difficulty=round(initial_fsrs_diff, 2),
                next_review_due=None,
                diagnosed_misconceptions=[],
            )
            self._store[key] = state
            return state

    def get_state(self, student_id: str, concept_id: str) -> Optional[StudentConceptMastery]:
        """
        Retrieves existing state or None if not found.
        """
        key = self._make_key(student_id, concept_id)
        with self._lock:
            return self._store.get(key)

    def update_state(self, state: StudentConceptMastery) -> None:
        """
        Updates or inserts student state in-memory and persists if configured.
        """
        key = self._make_key(state.student_id, state.concept_id)
        with self._lock:
            self._store[key] = state

        if self.persistence_filepath:
            self.save_to_file(self.persistence_filepath)

    def get_student_profile(self, student_id: str) -> List[StudentConceptMastery]:
        """
        Retrieves all concept states for a given student.
        """
        with self._lock:
            return [
                state
                for key, state in self._store.items()
                if key.startswith(f"{student_id}::")
            ]

    def get_weak_concepts(
        self, student_id: str, threshold: float = 0.50
    ) -> List[StudentConceptMastery]:
        """
        Returns concepts where student mastery is below threshold.
        """
        profile = self.get_student_profile(student_id)
        return [s for s in profile if s.bkt_p_mastery < threshold]

    def get_mastered_concepts(
        self, student_id: str, threshold: float = 0.95
    ) -> List[StudentConceptMastery]:
        """
        Returns concepts where student mastery meets or exceeds threshold.
        """
        profile = self.get_student_profile(student_id)
        return [s for s in profile if s.bkt_p_mastery >= threshold]

    def get_mastery_summary(self, student_id: str) -> Dict[str, Any]:
        """
        Computes aggregate metrics for student dashboard.
        """
        profile = self.get_student_profile(student_id)
        if not profile:
            return {
                "student_id": student_id,
                "total_concepts": 0,
                "mastered_count": 0,
                "learning_count": 0,
                "unlearned_count": 0,
                "average_mastery": 0.0,
            }

        mastered = sum(1 for s in profile if s.bkt_p_mastery >= 0.95)
        learning = sum(1 for s in profile if 0.50 <= s.bkt_p_mastery < 0.95)
        unlearned = sum(1 for s in profile if s.bkt_p_mastery < 0.50)
        avg_mastery = sum(s.bkt_p_mastery for s in profile) / len(profile)

        return {
            "student_id": student_id,
            "total_concepts": len(profile),
            "mastered_count": mastered,
            "learning_count": learning,
            "unlearned_count": unlearned,
            "average_mastery": round(avg_mastery, 4),
        }

    def save_to_file(self, filepath: str) -> None:
        """
        Serializes the entire store into a JSON file.
        """
        with self._lock:
            data = {k: v.model_dump() for k, v in self._store.items()}

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_file(self, filepath: str) -> None:
        """
        Loads store state from a JSON file.
        """
        if not os.path.exists(filepath):
            return

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        with self._lock:
            self._store = {
                k: StudentConceptMastery.model_validate(v) for k, v in data.items()
            }
