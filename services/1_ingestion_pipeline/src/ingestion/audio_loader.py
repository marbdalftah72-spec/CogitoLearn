
# ==================================================
# CogitoLearn - Audio Ingestion Engine
# File: audio_loader.py
# WhisperX + Word-Level Alignment + Unified AST
# ==================================================

from pathlib import Path
from math import floor, ceil, isfinite
from typing import Optional, Any

import whisperx

from src.ingestion.fingerprinting import FingerprintingEngine

from src.ingestion.schemas import (
    MediaModality,
    ContentType,
    RawMediaStream,
    IngestedElement,
    UnifiedDocumentAST,
)


class AudioLoader:

    SUPPORTED_FORMATS = {
        ".wav",
        ".mp3",
        ".m4a",
        ".ogg",
        ".flac",
    }

    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
        batch_size: int = 4,
    ):

        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.batch_size = batch_size

        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.align_language = None

        self.fingerprinting_engine = FingerprintingEngine()

    # --------------------------------------------------
    # 1. VALIDATE AUDIO FILE
    # --------------------------------------------------

    def validate_file(self, file_path: str) -> Path:

        path = Path(file_path).expanduser()

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {path.suffix}"
            )

        if not path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                "The provided path must point to a file."
            )

        if path.stat().st_size == 0:
            raise ValueError(
                "Audio file is empty."
            )

        return path.resolve()

    # --------------------------------------------------
    # 2. LOAD WHISPERX MODEL
    # --------------------------------------------------

    def load_model(self):

        if self.model is None:

            self.model = whisperx.load_model(
                self.model_name,
                self.device,
                compute_type=self.compute_type,
                language=self.language,
            )

        return self.model

    # --------------------------------------------------
    # 3. TRANSCRIBE AUDIO
    # --------------------------------------------------

    def transcribe(self, file_path: str):

        path = self.validate_file(file_path)

        model = self.load_model()

        audio = whisperx.load_audio(str(path))

        if audio is None or len(audio) == 0:
            raise RuntimeError(
                "WhisperX could not decode the audio file."
            )

        result = model.transcribe(
            audio,
            batch_size=self.batch_size,
        )

        if not isinstance(result, dict):
            raise RuntimeError(
                "WhisperX returned an invalid transcription result."
            )

        if not result.get("segments"):
            raise RuntimeError(
                "No speech segments were detected in the audio."
            )

        return result, audio

    # --------------------------------------------------
    # 4. LOAD ALIGNMENT MODEL
    # --------------------------------------------------

    def load_alignment_model(self, language: str):

        if (
            self.align_model is None
            or self.align_language != language
        ):

            self.align_model, self.align_metadata = (
                whisperx.load_align_model(
                    language_code=language,
                    device=self.device,
                )
            )

            self.align_language = language

        return self.align_model, self.align_metadata

    # --------------------------------------------------
    # 5. WORD-LEVEL ALIGNMENT
    # --------------------------------------------------

    def align_transcription(self, result, audio):

        language = result.get("language")

        if not language:
            raise ValueError(
                "WhisperX did not detect the audio language."
            )

        align_model, align_metadata = (
            self.load_alignment_model(language)
        )

        aligned_result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        if not isinstance(aligned_result, dict):
            raise RuntimeError(
                "WhisperX returned an invalid alignment result."
            )

        return aligned_result

    # --------------------------------------------------
    # 6. CONVERT SECONDS TO MILLISECONDS
    # --------------------------------------------------

    @staticmethod
    def seconds_to_ms(value: Any, is_end: bool = False) -> int:

        try:
            seconds = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                "Timestamp must be a valid number."
            )

        if not isfinite(seconds) or seconds < 0:
            raise ValueError(
                "Timestamp must be finite and non-negative."
            )

        if is_end:
            return ceil(seconds * 1000)

        return floor(seconds * 1000)

    # --------------------------------------------------
    # 7. EXTRACT WORD TIMESTAMPS
    # --------------------------------------------------

    def extract_words(self, segment):

        words = []

        for word in segment.get("words", []):

            word_text = word.get("word", "").strip()

            start = word.get("start")
            end = word.get("end")

            if not word_text or start is None or end is None:
                continue

            try:
                start_ms = self.seconds_to_ms(start)
                end_ms = self.seconds_to_ms(
                    end,
                    is_end=True,
                )
            except ValueError:
                continue

            if end_ms <= start_ms:
                continue

            words.append({
                "word": word_text,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "score": word.get("score"),
            })

        return words

    # --------------------------------------------------
    # 8. CONVERT SEGMENTS TO AST ELEMENTS
    # --------------------------------------------------

    def build_elements(
        self,
        segments,
        fingerprint: str,
        language: Optional[str],
        alignment_status: str,
        source_name: str,
    ):

        elements = []

        for segment_index, segment in enumerate(segments):

            text = segment.get("text", "").strip()

            if not text:
                continue

            words = self.extract_words(segment)

            start = segment.get("start")
            end = segment.get("end")

            try:

                if start is None or end is None:
                    raise ValueError(
                        "Segment timestamps are unavailable."
                    )

                start_ms = self.seconds_to_ms(start)
                end_ms = self.seconds_to_ms(
                    end,
                    is_end=True,
                )

                if end_ms <= start_ms:
                    raise ValueError(
                        "Invalid segment timestamps."
                    )

            except ValueError:

                if not words:
                    continue

                start_ms = words[0]["start_ms"]
                end_ms = words[-1]["end_ms"]

            if end_ms <= start_ms:
                continue

            provenance_id = (
                f"{fingerprint}:audio-segment:{segment_index}"
            )

            element = IngestedElement(
                provenance_id=provenance_id,
                content=text,
                content_type=ContentType.ASR_SPAN,
                start_ms=start_ms,
                end_ms=end_ms,
                metadata={
                    "segment_index": segment_index,
                    "language": language,
                    "alignment_status": alignment_status,
                    "word_timestamps": words,
                    "word_count": len(words),
                    "source_audio": source_name,
                },
            )

            elements.append(element)

        return elements

    # --------------------------------------------------
    # 9. LOAD AUDIO INTO UNIFIED AST
    # --------------------------------------------------

    def load_audio(self, file_path: str) -> UnifiedDocumentAST:

        path = self.validate_file(file_path)

        # Calculate fingerprint before model inference.
        fingerprint = (
            self.fingerprinting_engine.calculate_sha256(
                str(path)
            )
        )

        file_size = path.stat().st_size

        # Transcription.
        result, audio = self.transcribe(str(path))

        language = result.get("language")

        # Alignment with safe fallback.
        alignment_status = "COMPLETED"

        try:

            aligned_result = self.align_transcription(
                result,
                audio,
            )

            segments = aligned_result.get("segments", [])

            if not segments:
                raise RuntimeError(
                    "Alignment returned no segments."
                )

        except Exception as error:

            print(
                "WARNING: Word alignment unavailable:",
                error,
            )

            # Preserve original transcription.
            segments = result.get("segments", [])

            alignment_status = "UNAVAILABLE"

        # Build AST elements.
        elements = self.build_elements(
            segments=segments,
            fingerprint=fingerprint,
            language=language,
            alignment_status=alignment_status,
            source_name=path.name,
        )

        # Source metadata.
        source = RawMediaStream(
            file_name=path.name,
            file_path=str(path),
            modality=MediaModality.AUDIO,
            file_size=file_size,
            sha256=fingerprint,
            metadata={
                "extension": path.suffix.lower(),
                "fingerprint_algorithm": "SHA-256",
            },
        )

        # Document metadata.
        document_metadata = {
            "parser": "WhisperX",
            "model_name": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": language,
            "alignment_status": alignment_status,
            "total_segments": len(elements),
            "total_words": sum(
                element.metadata.get("word_count", 0)
                for element in elements
            ),
            "fingerprint_algorithm": "SHA-256",
        }

        return UnifiedDocumentAST(
            source=source,
            elements=elements,
            metadata=document_metadata,
        )


# ==================================================
# BACKWARD-COMPATIBLE HELPER
# ==================================================

def load_audio(file_path: str) -> UnifiedDocumentAST:

    loader = AudioLoader()

    return loader.load_audio(file_path)


# ==================================================
# MAIN EXECUTION
# ==================================================

if __name__ == "__main__":

    sample_path = "data/raw/sample_audio.wav"

    loader = AudioLoader()

    document = loader.load_audio(sample_path)

    print("\n========== AUDIO INGESTION SUMMARY ==========")

    print("File:", document.source.file_name)
    print("Modality:", document.source.modality)
    print("SHA256:", document.source.sha256)
    print("Language:", document.metadata.get("language"))
    print("Alignment:", document.metadata.get("alignment_status"))
    print("Total segments:", len(document.elements))
    print("Total words:", document.metadata.get("total_words"))

    for element in document.elements[:5]:

        print("\nText:", element.content)
        print("Start:", element.start_ms, "ms")
        print("End:", element.end_ms, "ms")
        print(
            "Word timestamps:",
            element.metadata.get("word_timestamps"),
        )
