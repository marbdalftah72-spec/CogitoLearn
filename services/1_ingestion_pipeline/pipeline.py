"""
Pipeline coordinator for Service 1: Ingestion.
"""
import uuid
from typing import BinaryIO
from common.contracts import UnifiedDocumentAST, MediaModality, EvidenceSpan

class IngestionPipeline:
    """Coordinates parsing, OCR, and speech transcription into UnifiedDocumentAST."""

    def __init__(self, gemini_api_key: str):
        self.api_key = gemini_api_key

    async def ingest_media(self, file_stream: BinaryIO, filename: str, course_id: str) -> UnifiedDocumentAST:
        """
        Receives raw media stream and produces a strongly typed UnifiedDocumentAST.
        """
        lecture_id = str(uuid.uuid4())
        ext = filename.split(".")[-1].lower()
        
        # Determine modality
        if ext in ["pdf", "docx"]:
            modality = MediaModality.PDF if ext == "pdf" else MediaModality.DOCX
        elif ext in ["mp3", "wav", "m4a"]:
            modality = MediaModality.AUDIO
        elif ext in ["mp4", "mkv"]:
            modality = MediaModality.VIDEO
        else:
            modality = MediaModality.SLIDES

        # TODO: Route to Docling or WhisperX accordingly
        return UnifiedDocumentAST(
            lecture_id=lecture_id,
            course_id=course_id,
            title=filename,
            modality=modality,
            total_spans=0,
            spans=[]
        )
