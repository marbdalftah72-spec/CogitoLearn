
# ==================================================
# CogitoLearn - Vision Processor Tests
# File: test_vision_processor.py
# ==================================================

import tempfile
import unittest

from pathlib import Path

from PIL import Image

from src.ingestion.vision_processor import VisionProcessor
from src.ingestion.schemas import (
    IngestedElement,
    ContentType,
)


class TestVisionProcessor(unittest.TestCase):

    def setUp(self):

        self.temp_dir = tempfile.TemporaryDirectory()

        self.image_path = (
            Path(self.temp_dir.name) / "sample.png"
        )

        # Create a real test image.
        image = Image.new(
            "RGB",
            (200, 100),
            color="white",
        )

        image.save(self.image_path)

        self.processor = VisionProcessor(
            enable_ocr=False
        )

    def tearDown(self):

        self.temp_dir.cleanup()

    # --------------------------------------------------
    # TEST 1: Validate image.
    # --------------------------------------------------

    def test_validate_image(self):

        result = self.processor.validate_image(
            self.image_path
        )

        self.assertTrue(result.exists())
        self.assertEqual(result.suffix, ".png")

    # --------------------------------------------------
    # TEST 2: Reject missing image.
    # --------------------------------------------------

    def test_missing_image(self):

        missing = (
            Path(self.temp_dir.name) / "missing.png"
        )

        with self.assertRaises(FileNotFoundError):

            self.processor.validate_image(missing)

    # --------------------------------------------------
    # TEST 3: Reject unsupported format.
    # --------------------------------------------------

    def test_unsupported_format(self):

        invalid = (
            Path(self.temp_dir.name) / "sample.txt"
        )

        invalid.write_text(
            "not an image",
            encoding="utf-8",
        )

        with self.assertRaises(ValueError):

            self.processor.validate_image(invalid)

    # --------------------------------------------------
    # TEST 4: Reject empty image.
    # --------------------------------------------------

    def test_empty_image(self):

        empty = (
            Path(self.temp_dir.name) / "empty.png"
        )

        empty.write_bytes(b"")

        with self.assertRaises(ValueError):

            self.processor.validate_image(empty)

    # --------------------------------------------------
    # TEST 5: Reject corrupted image.
    # --------------------------------------------------

    def test_corrupted_image(self):

        corrupted = (
            Path(self.temp_dir.name) / "corrupted.png"
        )

        corrupted.write_bytes(
            b"this is not a real image"
        )

        with self.assertRaises(ValueError):

            self.processor.process_image(corrupted)

    # --------------------------------------------------
    # TEST 6: Load image.
    # --------------------------------------------------

    def test_load_image(self):

        result = self.processor.load_image(
            self.image_path
        )

        image_rgb, width, height, image_format, mode = result

        self.assertEqual(width, 200)
        self.assertEqual(height, 100)
        self.assertEqual(image_format, "PNG")
        self.assertEqual(image_rgb.shape, (100, 200, 3))

    # --------------------------------------------------
    # TEST 7: Process image.
    # --------------------------------------------------

    def test_process_image(self):

        metadata = self.processor.process_image(
            self.image_path
        )

        self.assertEqual(
            metadata["processing_status"],
            "SUCCESS",
        )

        self.assertEqual(metadata["width"], 200)
        self.assertEqual(metadata["height"], 100)

        self.assertIn("detected_contours", metadata)
        self.assertIn("edge_pixels", metadata)

        self.assertEqual(
            metadata["ocr_status"],
            "DISABLED",
        )

    # --------------------------------------------------
    # TEST 8: Check aspect ratio.
    # --------------------------------------------------

    def test_aspect_ratio(self):

        metadata = self.processor.process_image(
            self.image_path
        )

        self.assertEqual(
            metadata["aspect_ratio"],
            2.0,
        )

    # --------------------------------------------------
    # TEST 9: Process AST element.
    # --------------------------------------------------

    def test_process_element(self):

        element = IngestedElement(
            content="Extracted test image",
            content_type=ContentType.IMAGE,
            asset_path=str(self.image_path),
        )

        updated_element = self.processor.process_element(
            element
        )

        self.assertIn(
            "vision",
            updated_element.metadata,
        )

        self.assertEqual(
            updated_element.metadata["vision"]["processing_status"],
            "SUCCESS",
        )

        self.assertEqual(
            updated_element.content,
            element.content,
        )

    # --------------------------------------------------
    # TEST 10: Return unchanged element without asset.
    # --------------------------------------------------

    def test_element_without_asset(self):

        element = IngestedElement(
            content="Text element",
            content_type=ContentType.PROSE,
        )

        result = self.processor.process_element(element)

        self.assertEqual(
            result.content,
            element.content,
        )

        self.assertNotIn(
            "vision",
            result.metadata,
        )


if __name__ == "__main__":

    unittest.main(verbosity=2)
