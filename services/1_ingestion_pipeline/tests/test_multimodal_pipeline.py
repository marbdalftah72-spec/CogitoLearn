
import json
import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock

from src.ingestion.multimodal_pipeline import MultimodalPipeline
from src.ingestion.schemas import (
    UnifiedDocumentAST,
    RawMediaStream,
    IngestedElement,
    MediaModality,
    ContentType,
)


def make_ast(asset_path=None):

    element = IngestedElement(
        content="Sample content",
        content_type=ContentType.PROSE,
        asset_path=asset_path,
    )

    source = RawMediaStream(
        file_name="sample.pdf",
        file_path="sample.pdf",
        modality=MediaModality.PDF,
        file_size=100,
        sha256="a" * 64,
    )

    return UnifiedDocumentAST(
        source=source,
        elements=[element],
        metadata={},
    )


def make_pipeline():

    pipeline = MultimodalPipeline.__new__(
        MultimodalPipeline
    )

    pipeline.document_loader = MagicMock()
    pipeline.audio_loader = MagicMock()
    pipeline.video_loader = MagicMock()

    pipeline.vision_processor = MagicMock()
    pipeline.vision_processor.SUPPORTED_EXTENSIONS = {
        ".png", ".jpg", ".jpeg",
        ".bmp", ".tiff", ".webp",
    }

    return pipeline


class TestMultimodalPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)

        self.pipeline = make_pipeline()

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_file(self, name, content=b"test"):
        path = self.directory / name
        path.write_bytes(content)
        return path

    def test_validate_valid_pdf(self):

        path = self.create_file("sample.pdf")

        result = self.pipeline.validate_input(path)

        self.assertEqual(result.suffix, ".pdf")
        self.assertTrue(result.is_absolute())

    def test_validate_missing_file(self):

        path = self.directory / "missing.pdf"

        with self.assertRaises(FileNotFoundError):
            self.pipeline.validate_input(path)

    def test_validate_unsupported_extension(self):

        path = self.create_file("sample.txt")

        with self.assertRaises(ValueError):
            self.pipeline.validate_input(path)

    def test_validate_empty_file(self):

        path = self.create_file("empty.pdf", b"")

        with self.assertRaises(ValueError):
            self.pipeline.validate_input(path)

    def test_document_routing(self):

        ast = make_ast()
        self.pipeline.document_loader.load_document.return_value = ast

        result = self.pipeline._load_by_modality(
            Path("sample.pdf")
        )

        self.pipeline.document_loader.load_document.assert_called_once()
        self.assertIs(result, ast)

    def test_audio_routing(self):

        ast = make_ast()
        self.pipeline.audio_loader.load_audio.return_value = ast

        result = self.pipeline._load_by_modality(
            Path("sample.wav")
        )

        self.pipeline.audio_loader.load_audio.assert_called_once()
        self.assertIs(result, ast)

    def test_video_routing(self):

        ast = make_ast()
        self.pipeline.video_loader.load_video.return_value = ast

        result = self.pipeline._load_by_modality(
            Path("sample.mp4")
        )

        self.pipeline.video_loader.load_video.assert_called_once()
        self.assertIs(result, ast)

    def test_unsupported_routing(self):

        with self.assertRaises(ValueError):
            self.pipeline._load_by_modality(
                Path("sample.txt")
            )

    def test_process_element_without_asset(self):

        ast = make_ast()

        result = self.pipeline._process_visual_elements(ast)

        self.pipeline.vision_processor.process_element.assert_not_called()
        self.assertEqual(len(result.elements), 1)

    def test_process_visual_element(self):

        image_path = self.create_file("image.png")
        ast = make_ast(str(image_path))

        original_element = ast.elements[0]

        processed_element = original_element.model_copy(
            update={
                "metadata": {
                    "vision": {
                        "processing_status": "SUCCESS"
                    }
                }
            }
        )

        self.pipeline.vision_processor.process_element.return_value = (
            processed_element
        )

        result = self.pipeline._process_visual_elements(ast)

        self.pipeline.vision_processor.process_element.assert_called_once()
        self.assertEqual(
            result.elements[0].metadata["vision"]["processing_status"],
            "SUCCESS",
        )

    def test_vision_error_is_recorded(self):

        image_path = self.create_file("image.png")
        ast = make_ast(str(image_path))

        self.pipeline.vision_processor.process_element.side_effect = (
            RuntimeError("Image processing failed")
        )

        result = self.pipeline._process_visual_elements(ast)

        errors = result.metadata["vision_processing_errors"]

        self.assertEqual(len(errors), 1)
        self.assertIn("Image processing failed", errors[0]["error"])
        self.assertEqual(len(result.elements), 1)

    def test_ingest_complete_flow(self):

        path = self.create_file("sample.pdf")
        ast = make_ast()

        self.pipeline.document_loader.load_document.return_value = ast

        result = self.pipeline.ingest(path)

        self.assertIsInstance(result, UnifiedDocumentAST)
        self.pipeline.document_loader.load_document.assert_called_once()

    def test_ingest_many_success_and_failure(self):

        paths = [
            self.directory / "first.pdf",
            self.directory / "second.pdf",
        ]

        self.pipeline.ingest = MagicMock(
            side_effect=[
                make_ast(),
                RuntimeError("Test failure"),
            ]
        )

        results = self.pipeline.ingest_many(paths)

        self.assertEqual(results[0]["status"], "SUCCESS")
        self.assertEqual(results[1]["status"], "FAILED")
        self.assertIsNone(results[1]["document"])

    def test_save_ast(self):

        ast = make_ast()

        output_path = self.directory / "output" / "result.json"

        saved_path = self.pipeline.save_ast(
            ast,
            output_path,
        )

        self.assertTrue(saved_path.exists())

        data = json.loads(
            saved_path.read_text(encoding="utf-8")
        )

        self.assertIn("source", data)
        self.assertIn("elements", data)

    def test_get_summary(self):

        ast = make_ast()

        summary = self.pipeline.get_summary(ast)

        self.assertEqual(summary["file"], "sample.pdf")
        self.assertEqual(summary["total_elements"], 1)
        self.assertEqual(summary["element_counts"]["PROSE"], 1)
        self.assertEqual(summary["sha256"], "a" * 64)


if __name__ == "__main__":
    unittest.main()
