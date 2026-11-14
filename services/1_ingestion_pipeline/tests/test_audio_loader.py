
# ==================================================
# CogitoLearn - Audio Loader Tests
# File: test_audio_loader.py
# Purpose: Test audio ingestion and AST conversion.
# ==================================================

import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from src.ingestion.audio_loader import AudioLoader
from src.ingestion.schemas import (
    MediaModality,
    ContentType,
)


class TestAudioLoader(unittest.TestCase):

    def setUp(self):

        self.temp_dir = tempfile.TemporaryDirectory()

        self.audio_path = Path(self.temp_dir.name) / "sample.wav"

        # A non-empty placeholder file.
        # WhisperX is mocked, so real audio is not required here.
        self.audio_path.write_bytes(b"fake audio data")

        self.loader = AudioLoader(
            model_name="small",
            device="cpu",
            compute_type="int8",
        )

        self.sample_segments = [
            {
                "text": " Hello world",
                "start": 0.2,
                "end": 1.5,
                "words": [
                    {
                        "word": "Hello",
                        "start": 0.2,
                        "end": 0.7,
                        "score": 0.99,
                    },
                    {
                        "word": "world",
                        "start": 0.8,
                        "end": 1.5,
                        "score": 0.98,
                    },
                ],
            }
        ]

        self.sample_result = {
            "language": "en",
            "segments": self.sample_segments,
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    # --------------------------------------------------
    # TEST 1: Validate supported audio file.
    # --------------------------------------------------

    def test_validate_audio_file(self):

        result = self.loader.validate_file(
            str(self.audio_path)
        )

        self.assertTrue(result.exists())
        self.assertTrue(result.is_file())
        self.assertEqual(result.suffix, ".wav")

    # --------------------------------------------------
    # TEST 2: Reject unsupported extension.
    # --------------------------------------------------

    def test_unsupported_format(self):

        invalid_path = Path(self.temp_dir.name) / "sample.txt"

        invalid_path.write_text("not audio", encoding="utf-8")

        with self.assertRaises(ValueError):

            self.loader.validate_file(str(invalid_path))

    # --------------------------------------------------
    # TEST 3: Reject missing file.
    # --------------------------------------------------

    def test_missing_file(self):

        missing_path = Path(self.temp_dir.name) / "missing.wav"

        with self.assertRaises(FileNotFoundError):

            self.loader.validate_file(str(missing_path))

    # --------------------------------------------------
    # TEST 4: Reject empty audio file.
    # --------------------------------------------------

    def test_empty_file(self):

        empty_path = Path(self.temp_dir.name) / "empty.wav"

        empty_path.write_bytes(b"")

        with self.assertRaises(ValueError):

            self.loader.validate_file(str(empty_path))

    # --------------------------------------------------
    # TEST 5: Convert seconds to milliseconds.
    # --------------------------------------------------

    def test_seconds_to_ms(self):

        self.assertEqual(
            self.loader.seconds_to_ms(1.234),
            1234,
        )

        self.assertEqual(
            self.loader.seconds_to_ms(1.234, is_end=True),
            1234,
        )

    # --------------------------------------------------
    # TEST 6: Reject invalid timestamps.
    # --------------------------------------------------

    def test_invalid_timestamp(self):

        with self.assertRaises(ValueError):

            self.loader.seconds_to_ms(-1)

        with self.assertRaises(ValueError):

            self.loader.seconds_to_ms("invalid")

    # --------------------------------------------------
    # TEST 7: Extract word timestamps.
    # --------------------------------------------------

    def test_extract_words(self):

        words = self.loader.extract_words(
            self.sample_segments[0]
        )

        self.assertEqual(len(words), 2)

        self.assertEqual(words[0]["word"], "Hello")
        self.assertEqual(words[0]["start_ms"], 200)
        self.assertEqual(words[0]["end_ms"], 700)

        self.assertEqual(words[1]["word"], "world")
        self.assertEqual(words[1]["start_ms"], 800)
        self.assertEqual(words[1]["end_ms"], 1500)

    # --------------------------------------------------
    # TEST 8: Build AST elements.
    # --------------------------------------------------

    def test_build_elements(self):

        elements = self.loader.build_elements(
            segments=self.sample_segments,
            fingerprint="a" * 64,
            language="en",
            alignment_status="COMPLETED",
            source_name="sample.wav",
        )

        self.assertEqual(len(elements), 1)

        element = elements[0]

        self.assertEqual(element.content, "Hello world")
        self.assertEqual(element.content_type, ContentType.ASR_SPAN)

        self.assertEqual(element.start_ms, 200)
        self.assertEqual(element.end_ms, 1500)

        self.assertTrue(element.provenance_id)

        self.assertEqual(
            element.metadata["word_count"],
            2,
        )

    # --------------------------------------------------
    # TEST 9: Complete audio ingestion with mocked models.
    # --------------------------------------------------

    def test_load_audio_success(self):

        with (
            patch.object(
                self.loader,
                "transcribe",
                return_value=(self.sample_result, object()),
            ),
            patch.object(
                self.loader,
                "align_transcription",
                return_value={
                    "segments": self.sample_segments,
                },
            ),
        ):

            document = self.loader.load_audio(
                str(self.audio_path)
            )

        self.assertEqual(
            document.source.modality,
            MediaModality.AUDIO,
        )

        self.assertEqual(
            document.source.file_name,
            "sample.wav",
        )

        self.assertRegex(
            document.source.sha256,
            r"^[a-fA-F0-9]{64}$",
        )

        self.assertEqual(len(document.elements), 1)

        self.assertEqual(
            document.metadata["alignment_status"],
            "COMPLETED",
        )

        self.assertEqual(
            document.metadata["language"],
            "en",
        )

        self.assertEqual(
            document.metadata["total_words"],
            2,
        )

    # --------------------------------------------------
    # TEST 10: Preserve transcription if alignment fails.
    # --------------------------------------------------

    def test_alignment_fallback(self):

        with (
            patch.object(
                self.loader,
                "transcribe",
                return_value=(self.sample_result, object()),
            ),
            patch.object(
                self.loader,
                "align_transcription",
                side_effect=RuntimeError("Alignment unavailable"),
            ),
        ):

            document = self.loader.load_audio(
                str(self.audio_path)
            )

        self.assertEqual(len(document.elements), 1)

        self.assertEqual(
            document.elements[0].content,
            "Hello world",
        )

        self.assertEqual(
            document.metadata["alignment_status"],
            "UNAVAILABLE",
        )

    # --------------------------------------------------
    # TEST 11: Check document metadata.
    # --------------------------------------------------

    def test_document_metadata(self):

        with (
            patch.object(
                self.loader,
                "transcribe",
                return_value=(self.sample_result, object()),
            ),
            patch.object(
                self.loader,
                "align_transcription",
                return_value={
                    "segments": self.sample_segments,
                },
            ),
        ):

            document = self.loader.load_audio(
                str(self.audio_path)
            )

        self.assertEqual(
            document.metadata["parser"],
            "WhisperX",
        )

        self.assertEqual(
            document.metadata["model_name"],
            "small",
        )

        self.assertEqual(
            document.metadata["total_segments"],
            1,
        )


if __name__ == "__main__":

    unittest.main(verbosity=2)
