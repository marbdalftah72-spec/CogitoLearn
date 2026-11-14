
# ==================================================
# CogitoLearn - Video Loader Tests
# File: tests/test_video_loader.py
# ==================================================

import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock, patch

from src.ingestion.video_loader import VideoLoader
from src.ingestion.schemas import (
    ContentType,
    IngestedElement,
    MediaModality,
    UnifiedDocumentAST,
    RawMediaStream,
)


class FakeVideoCapture:

    def __init__(self, video_path):
        self.released = False
        self.current_timestamp = 0

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == 5:  # cv2.CAP_PROP_FPS
            return 10.0

        if prop == 7:  # cv2.CAP_PROP_FRAME_COUNT
            return 30.0

        return 0

    def set(self, prop, value):
        self.current_timestamp = value
        return True

    def read(self):
        if self.current_timestamp < 3000:
            return True, "fake_frame"

        return False, None

    def release(self):
        self.released = True


class TestVideoLoader(unittest.TestCase):

    def setUp(self):
        self.loader = VideoLoader(
            frame_interval_seconds=1,
            audio_loader=Mock(),
        )

    # ==============================================
    # 1. INITIALIZATION
    # ==============================================

    def test_default_frame_interval(self):

        loader = VideoLoader(
            audio_loader=Mock()
        )

        self.assertEqual(
            loader.frame_interval_seconds,
            10,
        )

    def test_invalid_frame_interval_zero(self):

        with self.assertRaises(ValueError):
            VideoLoader(
                frame_interval_seconds=0,
                audio_loader=Mock(),
            )

    def test_invalid_frame_interval_negative(self):

        with self.assertRaises(ValueError):
            VideoLoader(
                frame_interval_seconds=-1,
                audio_loader=Mock(),
            )

    def test_invalid_frame_interval_float(self):

        with self.assertRaises(ValueError):
            VideoLoader(
                frame_interval_seconds=1.5,
                audio_loader=Mock(),
            )

    # ==============================================
    # 2. VIDEO VALIDATION
    # ==============================================

    def test_missing_video(self):

        with self.assertRaises(FileNotFoundError):
            self.loader._validate_video(
                Path("missing_video.mp4")
            )

    def test_unsupported_video_format(self):

        with tempfile.TemporaryDirectory() as temp_dir:

            video_path = Path(temp_dir) / "video.txt"
            video_path.write_bytes(b"video")

            with self.assertRaises(ValueError):
                self.loader._validate_video(video_path)

    def test_empty_video(self):

        with tempfile.TemporaryDirectory() as temp_dir:

            video_path = Path(temp_dir) / "video.mp4"
            video_path.write_bytes(b"")

            with self.assertRaises(ValueError):
                self.loader._validate_video(video_path)

    def test_valid_video(self):

        with tempfile.TemporaryDirectory() as temp_dir:

            video_path = Path(temp_dir) / "video.mp4"
            video_path.write_bytes(b"fake video")

            result = self.loader._validate_video(video_path)

            self.assertTrue(result.exists())
            self.assertEqual(result.suffix, ".mp4")

    # ==============================================
    # 3. SHA256
    # ==============================================

    def test_calculate_sha256(self):

        import hashlib

        with tempfile.TemporaryDirectory() as temp_dir:

            video_path = Path(temp_dir) / "video.mp4"
            video_path.write_bytes(b"CogitoLearn video")

            result = self.loader._calculate_sha256(video_path)

            expected = hashlib.sha256(
                b"CogitoLearn video"
            ).hexdigest()

            self.assertEqual(result, expected)
            self.assertEqual(len(result), 64)

    # ==============================================
    # 4. FFMPEG VALIDATION
    # ==============================================

    @patch("src.ingestion.video_loader.shutil.which")
    def test_missing_ffmpeg_tools(self, mock_which):

        mock_which.return_value = None

        with self.assertRaises(RuntimeError):
            self.loader._check_ffmpeg_tools()

    @patch("src.ingestion.video_loader.shutil.which")
    def test_ffmpeg_tools_available(self, mock_which):

        mock_which.return_value = "C:/ffmpeg/bin/tool.exe"

        self.loader._check_ffmpeg_tools()

        self.assertEqual(mock_which.call_count, 2)

    # ==============================================
    # 5. FRAME EXTRACTION
    # ==============================================

    @patch("src.ingestion.video_loader.cv2.imwrite")
    @patch("src.ingestion.video_loader.cv2.VideoCapture")
    def test_extract_frames(
        self,
        mock_capture,
        mock_imwrite,
    ):

        mock_capture.return_value = FakeVideoCapture("video.mp4")
        mock_imwrite.return_value = True

        with tempfile.TemporaryDirectory() as temp_dir:

            video_path = Path(temp_dir) / "video.mp4"
            video_path.write_bytes(b"fake video")

            output_dir = Path(temp_dir) / "frames"

            elements = self.loader.extract_frames(
                video_path,
                output_dir,
                video_sha256="a" * 64,
            )

            self.assertEqual(len(elements), 3)

            self.assertTrue(
                all(
                    element.content_type == ContentType.VIDEO_FRAME
                    for element in elements
                )
            )

            self.assertEqual(
                [element.start_ms for element in elements],
                [0, 1000, 2000],
            )

            self.assertEqual(
                [element.end_ms for element in elements],
                [1, 1001, 2001],
            )

            self.assertTrue(
                all(element.asset_path for element in elements)
            )

            self.assertTrue(
                all(
                    element.provenance_id.startswith("a" * 64)
                    for element in elements
                )
            )

    @patch("src.ingestion.video_loader.cv2.VideoCapture")
    def test_video_cannot_be_opened(self, mock_capture):

        capture = Mock()
        capture.isOpened.return_value = False
        mock_capture.return_value = capture

        with tempfile.TemporaryDirectory() as temp_dir:

            video_path = Path(temp_dir) / "video.mp4"
            video_path.write_bytes(b"fake video")

            with self.assertRaises(RuntimeError):
                self.loader.extract_frames(
                    video_path,
                    Path(temp_dir) / "frames",
                )

            capture.release.assert_called_once()

    @patch("src.ingestion.video_loader.cv2.VideoCapture")
    def test_invalid_video_properties(self, mock_capture):

        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 0
        mock_capture.return_value = capture

        with tempfile.TemporaryDirectory() as temp_dir:

            video_path = Path(temp_dir) / "video.mp4"
            video_path.write_bytes(b"fake video")

            with self.assertRaises(RuntimeError):
                self.loader.extract_frames(
                    video_path,
                    Path(temp_dir) / "frames",
                )

            capture.release.assert_called_once()

    @patch("src.ingestion.video_loader.cv2.imwrite")
    @patch("src.ingestion.video_loader.cv2.VideoCapture")
    def test_frame_save_failure(
        self,
        mock_capture,
        mock_imwrite,
    ):

        mock_capture.return_value = FakeVideoCapture("video.mp4")
        mock_imwrite.return_value = False

        with tempfile.TemporaryDirectory() as temp_dir:

            video_path = Path(temp_dir) / "video.mp4"
            video_path.write_bytes(b"fake video")

            with self.assertRaises(RuntimeError):
                self.loader.extract_frames(
                    video_path,
                    Path(temp_dir) / "frames",
                )

    # ==============================================
    # 6. VIDEO INGESTION WITHOUT AUDIO
    # ==============================================

    @patch.object(VideoLoader, "_has_audio_stream", return_value=False)
    @patch.object(VideoLoader, "extract_frames")
    def test_load_video_without_audio(
        self,
        mock_extract_frames,
        mock_has_audio,
    ):

        frame = IngestedElement(
            content="Video frame at 0.00 seconds",
            content_type=ContentType.VIDEO_FRAME,
            start_ms=0,
            end_ms=1,
            asset_path="frame.jpg",
        )

        mock_extract_frames.return_value = [frame]

        with tempfile.TemporaryDirectory() as temp_dir:

            video_path = Path(temp_dir) / "video.mp4"
            video_path.write_bytes(b"fake video")

            result = self.loader.load_video(str(video_path))

            self.assertIsInstance(result, UnifiedDocumentAST)
            self.assertEqual(result.source.modality, MediaModality.VIDEO)
            self.assertEqual(len(result.elements), 1)
            self.assertFalse(result.metadata["has_audio"])
            self.assertEqual(result.metadata["total_frames"], 1)
            self.assertEqual(result.metadata["total_audio_segments"], 0)

            mock_has_audio.assert_called_once()

    # ==============================================
    # 7. VIDEO INGESTION WITH AUDIO
    # ==============================================

    @patch.object(VideoLoader, "_has_audio_stream", return_value=True)
    @patch.object(VideoLoader, "extract_audio")
    @patch.object(VideoLoader, "extract_frames")
    def test_load_video_with_audio(
        self,
        mock_extract_frames,
        mock_extract_audio,
        mock_has_audio,
    ):

        frame = IngestedElement(
            content="Video frame at 0.00 seconds",
            content_type=ContentType.VIDEO_FRAME,
            start_ms=0,
            end_ms=1,
            asset_path="frame.jpg",
        )

        audio_element = IngestedElement(
            content="Hello CogitoLearn",
            content_type=ContentType.ASR_SPAN,
            start_ms=0,
            end_ms=1200,
        )

        audio_source = RawMediaStream(
            file_name="audio.wav",
            file_path="audio.wav",
            modality=MediaModality.AUDIO,
        )

        audio_ast = UnifiedDocumentAST(
            source=audio_source,
            elements=[audio_element],
        )

        self.loader.audio_loader.load_audio.return_value = audio_ast

        mock_extract_frames.return_value = [frame]

        with tempfile.TemporaryDirectory() as temp_dir:

            video_path = Path(temp_dir) / "video.mp4"
            video_path.write_bytes(b"fake video")

            result = self.loader.load_video(str(video_path))

            self.assertEqual(len(result.elements), 2)
            self.assertTrue(result.metadata["has_audio"])
            self.assertEqual(result.metadata["total_frames"], 1)
            self.assertEqual(result.metadata["total_audio_segments"], 1)

            updated_audio = result.elements[1]

            self.assertEqual(
                updated_audio.content_type,
                ContentType.ASR_SPAN,
            )

            self.assertTrue(
                updated_audio.metadata["original_video_audio"]
            )

            self.assertEqual(
                updated_audio.metadata["source_modality"],
                "VIDEO",
            )

            self.loader.audio_loader.load_audio.assert_called_once()

    # ==============================================
    # 8. PROVENANCE
    # ==============================================

    def test_frame_provenance_format(self):

        element = IngestedElement(
            provenance_id="abc123:video-frame:0",
            content="Video frame at 0.00 seconds",
            content_type=ContentType.VIDEO_FRAME,
            start_ms=0,
            end_ms=1,
        )

        self.assertIn(
            "video-frame",
            element.provenance_id,
        )

    def test_frame_requires_valid_time_interval(self):

        with self.assertRaises(ValueError):
            IngestedElement(
                content="Invalid frame",
                content_type=ContentType.VIDEO_FRAME,
                start_ms=1000,
            )


if __name__ == "__main__":
    unittest.main()
