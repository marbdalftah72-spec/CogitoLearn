
# ==================================================
# CogitoLearn - Video Ingestion Engine
# File: video_loader.py
# Purpose:
# Extract video frames and audio,
# transcribe speech, and preserve provenance.
# ==================================================

import hashlib
import json
import math
import shutil
import subprocess
import tempfile

from pathlib import Path
from typing import Optional

import cv2

from src.ingestion.schemas import (
    ContentType,
    IngestedElement,
    MediaModality,
    RawMediaStream,
    UnifiedDocumentAST,
)

from src.ingestion.audio_loader import AudioLoader


class VideoLoader:

    SUPPORTED_FORMATS = {
        ".mp4", ".mov", ".mkv", ".avi",
        ".webm", ".mpeg", ".mpg", ".m4v",
    }

    def __init__(
        self,
        frame_interval_seconds: int = 10,
        audio_loader=None,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ):

        if (
            isinstance(frame_interval_seconds, bool)
            or not isinstance(frame_interval_seconds, int)
            or frame_interval_seconds <= 0
        ):
            raise ValueError(
                "Frame interval must be a positive integer."
            )

        self.frame_interval_seconds = frame_interval_seconds

        self.audio_loader = audio_loader or AudioLoader(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
        )

    # ==================================================
    # 1. VALIDATE VIDEO
    # ==================================================

    def _validate_video(self, video_path: Path) -> Path:

        video_path = Path(video_path).expanduser()

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        if not video_path.is_file():
            raise ValueError(
                "Path must point to a file."
            )

        if video_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported video format: {video_path.suffix}"
            )

        if video_path.stat().st_size == 0:
            raise ValueError(
                "Video file is empty."
            )

        return video_path.resolve()

    # ==================================================
    # 2. CALCULATE SHA256
    # ==================================================

    @staticmethod
    def _calculate_sha256(file_path: Path) -> str:

        sha256 = hashlib.sha256()

        with file_path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                sha256.update(chunk)

        return sha256.hexdigest()

    # ==================================================
    # 3. CHECK REQUIRED FFMPEG TOOLS
    # ==================================================

    @staticmethod
    def _check_ffmpeg_tools() -> None:

        missing_tools = [
            tool
            for tool in ("ffmpeg", "ffprobe")
            if shutil.which(tool) is None
        ]

        if missing_tools:
            raise RuntimeError(
                "Required FFmpeg tools not found: "
                + ", ".join(missing_tools)
                + ". Install FFmpeg and add it to PATH."
            )

    # ==================================================
    # 4. CHECK AUDIO STREAM
    # ==================================================

    def _has_audio_stream(self, video_path: Path) -> bool:

        self._check_ffmpeg_tools()

        command = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=index",
            "-of", "json",
            str(video_path),
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )

            metadata = json.loads(result.stdout)

        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Could not inspect video audio stream: {exc.stderr}"
            ) from exc

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "FFprobe returned invalid JSON."
            ) from exc

        return bool(metadata.get("streams"))

    # ==================================================
    # 5. EXTRACT VIDEO FRAMES
    # ==================================================

    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        video_sha256: Optional[str] = None,
    ) -> list[IngestedElement]:

        video_path = Path(video_path)
        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if video_sha256 is None:
            video_sha256 = self._calculate_sha256(video_path)

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                "Could not open video file."
            )

        elements = []

        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )

            if (
                not math.isfinite(fps)
                or not math.isfinite(frame_count)
                or fps <= 0
                or frame_count <= 0
            ):
                raise RuntimeError(
                    "Invalid video duration or frame rate."
                )

            duration_ms = (frame_count / fps) * 1000

            duration_limit_ms = math.ceil(duration_ms)

            interval_ms = self.frame_interval_seconds * 1000

            timestamp_ms = 0
            frame_index = 0

            while timestamp_ms < duration_ms:

                capture.set(
                    cv2.CAP_PROP_POS_MSEC,
                    timestamp_ms,
                )

                success, frame = capture.read()

                if not success:
                    break

                frame_name = (
                    f"frame_{frame_index:04d}.jpg"
                )

                frame_path = output_dir / frame_name

                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(
                        f"Could not save frame: {frame_path}"
                    )

                # The schema requires start_ms and end_ms together.
                # A frame is represented as a 1 ms interval.

                end_ms = min(
                    timestamp_ms + 1,
                    duration_limit_ms,
                )

                element = IngestedElement(
                    provenance_id=(
                        f"{video_sha256}:video-frame:{frame_index}"
                    ),

                    content=(
                        f"Video frame at "
                        f"{timestamp_ms / 1000:.2f} seconds"
                    ),

                    content_type=ContentType.VIDEO_FRAME,

                    start_ms=timestamp_ms,
                    end_ms=end_ms,

                    asset_path=str(frame_path.resolve()),

                    metadata={
                        "source_video": str(video_path.resolve()),
                        "source_sha256": video_sha256,
                        "frame_index": frame_index,
                        "timestamp_ms": timestamp_ms,
                        "timestamp_seconds": round(
                            timestamp_ms / 1000,
                            3,
                        ),
                        "frame_interval_seconds": (
                            self.frame_interval_seconds
                        ),
                        "frame_representation": "1ms_interval",
                    },
                )

                elements.append(element)

                frame_index += 1
                timestamp_ms += interval_ms

        finally:
            capture.release()

        if not elements:
            raise RuntimeError(
                "No video frames could be extracted."
            )

        return elements

    # ==================================================
    # 6. EXTRACT AUDIO
    # ==================================================

    def extract_audio(
        self,
        video_path: Path,
        output_path: Path,
    ) -> Path:

        self._check_ffmpeg_tools()

        video_path = Path(video_path)
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(output_path),
        ]

        try:
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )

        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Audio extraction failed: {exc.stderr}"
            ) from exc

        if (
            not output_path.exists()
            or output_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "Audio extraction produced an empty file."
            )

        return output_path

    # ==================================================
    # 7. LOAD VIDEO
    # ==================================================

    def load_video(
        self,
        file_path: str,
    ) -> UnifiedDocumentAST:

        video_path = self._validate_video(
            Path(file_path)
        )

        video_sha256 = self._calculate_sha256(
            video_path
        )

        output_dir = (
            Path("data/processed/extracted_frames")
            / video_path.stem
        )

        # A. Extract video frames

        elements = self.extract_frames(
            video_path=video_path,
            output_dir=output_dir,
            video_sha256=video_sha256,
        )

        # B. Extract and transcribe audio when available

        has_audio = self._has_audio_stream(video_path)

        if has_audio:

            with tempfile.TemporaryDirectory() as temp_dir:

                audio_path = (
                    Path(temp_dir) / "extracted_audio.wav"
                )

                self.extract_audio(
                    video_path,
                    audio_path,
                )

                audio_ast = self.audio_loader.load_audio(
                    str(audio_path)
                )

                for index, audio_element in enumerate(
                    audio_ast.elements
                ):

                    metadata = dict(
                        audio_element.metadata or {}
                    )

                    metadata.update({
                        "source_video": str(video_path),
                        "source_video_sha256": video_sha256,
                        "source_modality": MediaModality.VIDEO.value,
                        "original_video_audio": True,
                        "original_audio_provenance_id": (
                            audio_element.provenance_id
                        ),
                    })

                    updated_element = audio_element.model_copy(
                        update={
                            "provenance_id": (
                                f"{video_sha256}:audio-segment:{index}"
                            ),
                            "metadata": metadata,
                        }
                    )

                    elements.append(updated_element)

        # C. Create unified video source

        source = RawMediaStream(
            file_name=video_path.name,
            file_path=str(video_path),
            modality=MediaModality.VIDEO,
            file_size=video_path.stat().st_size,
            sha256=video_sha256,
            metadata={
                "has_audio": has_audio,
                "frame_interval_seconds": (
                    self.frame_interval_seconds
                ),
                "video_sha256": video_sha256,
            },
        )

        # D. Return unified AST

        return UnifiedDocumentAST(
            source=source,
            elements=elements,
            metadata={
                "ingestion_engine": "VideoLoader",
                "total_frames": sum(
                    element.content_type == ContentType.VIDEO_FRAME
                    for element in elements
                ),
                "total_audio_segments": sum(
                    element.content_type == ContentType.ASR_SPAN
                    for element in elements
                ),
                "has_audio": has_audio,
            },
        )


# ==================================================
# 8. BASIC EXECUTION
# ==================================================

if __name__ == "__main__":

    video_file = Path(
        "data/raw/sample_lecture.mp4"
    )

    loader = VideoLoader(
        frame_interval_seconds=10
    )

    result = loader.load_video(
        str(video_file)
    )

    output_path = Path(
        "data/processed/video_test.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    video_frames = sum(
        item.content_type == ContentType.VIDEO_FRAME
        for item in result.elements
    )

    audio_segments = sum(
        item.content_type == ContentType.ASR_SPAN
        for item in result.elements
    )

    print("\n========== VIDEO INGESTION ==========")
    print("Source:", result.source.file_name)
    print("Total elements:", len(result.elements))
    print("Video frames:", video_frames)
    print("Audio segments:", audio_segments)
    print("Has audio:", result.source.metadata["has_audio"])
    print("Output:", output_path)
    print("\nVIDEO INGESTION COMPLETED!")
