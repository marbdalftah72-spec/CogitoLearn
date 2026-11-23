from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class VectorPayload(BaseModel):
    chunk_id: int
    text: str
    lecture_id: Optional[str] = "default_lecture"
    course_id: Optional[str] = "nlp_course"
    hierarchy_level: int = 0
    modality_kind: str = "PROSE"
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)
