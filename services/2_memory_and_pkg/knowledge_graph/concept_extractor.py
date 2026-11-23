import re
from typing import List
from common.contracts import PKGConcept, ConceptType

class ConceptExtractor:
    """
    Extracts pedagogical concepts from lecture text and notes.
    """
    def __init__(self, course_id: str = "nlp_course"):
        self.course_id = course_id

    def extract_concepts(self, text: str) -> List[PKGConcept]:
        candidates = [
            ("Natural Language Processing", "معالجة اللغات الطبيعية", ConceptType.DEFINITION),
            ("NLU", "فهم اللغة الطبيعية", ConceptType.DEFINITION),
            ("NLG", "توليد اللغة الطبيعية", ConceptType.DEFINITION),
            ("Tokenization", "التقطيع اللغوي", ConceptType.ALGORITHM),
            ("Stemming", "التجذير", ConceptType.ALGORITHM),
            ("Lemmatization", "الرد للأصل المعجمي", ConceptType.ALGORITHM),
            ("TF-IDF", "تكرار المصطلح مقابل مقلوب تكرار الوثيقة", ConceptType.FORMULA),
            ("N-Gram", "نموذج الإن-جرام", ConceptType.ALGORITHM),
            ("Bag of Words", "حقيبة الكلمات", ConceptType.DEFINITION),
            ("One-Hot Encoding", "الترميز الأحادي", ConceptType.ALGORITHM),
        ]
        
        extracted = []
        for name_en, name_ar, c_type in candidates:
            if re.search(r'\b' + re.escape(name_en) + r'\b', text, re.IGNORECASE):
                cid = name_en.lower().replace(" ", "_").replace("-", "_")
                concept = PKGConcept(
                    concept_id=cid,
                    course_id=self.course_id,
                    name_ar=name_ar,
                    name_en=name_en,
                    concept_type=c_type,
                    base_difficulty=0.5
                )
                extracted.append(concept)
        return extracted
