
# ==================================================
# CogitoLearn - Unified Data Contracts
# File: schemas.py
# Purpose:
# Define unified Pydantic v2 schemas for all
# multimodal ingestion inputs and outputs.
# ==================================================

from enum import Enum
from typing import Optional, Any
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ==================================================
# 1. MEDIA MODALITIES
# ==================================================

class MediaModality(str, Enum):

    PDF = "PDF"
    DOCX = "DOCX"
    PPTX = "PPTX"
    AUDIO = "AUDIO"
    VIDEO = "VIDEO"
    IMAGE = "IMAGE"


# ==================================================
# 2. CONTENT TYPES
# ==================================================

class ContentType(str, Enum):

    PROSE = "PROSE"
    TABLE_MD = "TABLE_MD"
    LATEX_EQ = "LATEX_EQ"
    DIAGRAM_AST = "DIAGRAM_AST"
    ASR_SPAN = "ASR_SPAN"
    IMAGE = "IMAGE"
    AUDIO_TRANSCRIPT = "AUDIO_TRANSCRIPT"
    VIDEO_FRAME = "VIDEO_FRAME"
    DIAGRAM_DESCRIPTION = "DIAGRAM_DESCRIPTION"


# ==================================================
# 3. BASE CONFIGURATION
# ==================================================

class IngestionBaseModel(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


# ==================================================
# 4. BOUNDING BOX
# ==================================================

class BoundingBox(IngestionBaseModel):

    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(gt=0)
    y1: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_coordinates(self):

        if self.x1 <= self.x0:
            raise ValueError(
                "x1 must be greater than x0."
            )

        if self.y1 <= self.y0:
            raise ValueError(
                "y1 must be greater than y0."
            )

        return self


# ==================================================
# 5. RAW MEDIA INPUT CONTRACT
# ==================================================

class RawMediaStream(IngestionBaseModel):

    file_name: str = Field(min_length=1)

    file_path: str = Field(min_length=1)

    modality: MediaModality

    file_size: Optional[int] = Field(
        default=None,
        ge=0,
    )

    sha256: Optional[str] = Field(
        default=None,
        pattern=r"^[a-fA-F0-9]{64}$",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ==================================================
# 6. INGESTED ELEMENT CONTRACT
# ==================================================

class IngestedElement(IngestionBaseModel):

    element_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
    )

    provenance_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
    )

    content: str = Field(min_length=1)

    content_type: ContentType

    page_no: Optional[int] = Field(
        default=None,
        ge=1,
    )

    bbox: Optional[BoundingBox] = None

    asset_path: Optional[str] = Field(
        default=None,
        min_length=1,
    )

    start_ms: Optional[int] = Field(
        default=None,
        ge=0,
    )

    end_ms: Optional[int] = Field(
        default=None,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_provenance_and_timing(self):

        if (
            self.start_ms is None
        ) != (
            self.end_ms is None
        ):
            raise ValueError(
                "start_ms and end_ms must be provided together."
            )

        if (
            self.start_ms is not None
            and self.end_ms <= self.start_ms
        ):
            raise ValueError(
                "end_ms must be greater than start_ms."
            )

        return self


# ==================================================
# 7. UNIFIED DOCUMENT AST
# ==================================================

class UnifiedDocumentAST(IngestionBaseModel):

    document_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
    )

    source: RawMediaStream

    elements: list[IngestedElement] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_document(self):

        element_ids = [
            element.element_id
            for element in self.elements
        ]

        if len(element_ids) != len(set(element_ids)):
            raise ValueError(
                "Duplicate element_id values are not allowed."
            )

        return self
