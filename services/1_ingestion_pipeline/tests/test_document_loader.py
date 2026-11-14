
# ==================================================
# CogitoLearn - Document Loader Tests
# File: test_document_loader.py
# Purpose: Test PDF ingestion and AST validation.
# ==================================================

import re
import unittest

from pathlib import Path

from src.ingestion.document_loader import PDFDocumentLoader
from src.ingestion.schemas import MediaModality


# ==================================================
# TEST CONFIGURATION
# ==================================================

PDF_PATH = Path("data/raw/sample_lecture.pdf")


class TestDocumentLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.loader = PDFDocumentLoader()

    def load_sample(self):
        self.assertTrue(
            PDF_PATH.exists(),
            f"Sample PDF does not exist: {PDF_PATH.resolve()}"
        )

        return self.loader.load_pdf(str(PDF_PATH))

    # --------------------------------------------------
    # TEST 1: Check PDF existence.
    # --------------------------------------------------

    def test_pdf_exists(self):

        self.assertTrue(PDF_PATH.exists())
        self.assertTrue(PDF_PATH.is_file())

    # --------------------------------------------------
    # TEST 2: Check document loading.
    # --------------------------------------------------

    def test_document_loading(self):

        document = self.load_sample()

        self.assertIsNotNone(document)
        self.assertEqual(
            document.source.file_name,
            "sample_lecture.pdf"
        )
        self.assertGreater(len(document.elements), 0)

    # --------------------------------------------------
    # TEST 3: Check extracted content.
    # --------------------------------------------------

    def test_extracted_content(self):

        document = self.load_sample()

        for element in document.elements:

            self.assertIsInstance(element.content, str)

            self.assertTrue(
                element.content.strip(),
                "An empty content element was extracted."
            )

    # --------------------------------------------------
    # TEST 4: Check PDF modality.
    # --------------------------------------------------

    def test_source_modality(self):

        document = self.load_sample()

        self.assertEqual(
            document.source.modality,
            MediaModality.PDF
        )

    # --------------------------------------------------
    # TEST 5: Check SHA-256 fingerprint.
    # --------------------------------------------------

    def test_sha256_fingerprint(self):

        document = self.load_sample()

        fingerprint = document.source.sha256

        self.assertIsNotNone(fingerprint)

        self.assertTrue(
            re.fullmatch(r"[a-fA-F0-9]{64}", fingerprint)
        )

    # --------------------------------------------------
    # TEST 6: Check source metadata.
    # --------------------------------------------------

    def test_source_metadata(self):

        document = self.load_sample()

        self.assertGreater(document.source.file_size, 0)

        self.assertIsInstance(
            document.source.metadata,
            dict
        )

    # --------------------------------------------------
    # TEST 7: Check UnifiedDocumentAST.
    # --------------------------------------------------

    def test_unified_document_ast(self):

        document = self.load_sample()

        self.assertTrue(document.document_id)
        self.assertIsInstance(document.elements, list)
        self.assertIsInstance(document.metadata, dict)

    # --------------------------------------------------
    # TEST 8: Check element IDs.
    # --------------------------------------------------

    def test_element_identifiers(self):

        document = self.load_sample()

        element_ids = [
            element.element_id
            for element in document.elements
        ]

        for element in document.elements:

            self.assertTrue(element.element_id)
            self.assertTrue(element.provenance_id)

        self.assertEqual(
            len(element_ids),
            len(set(element_ids)),
            "Duplicate element IDs were found."
        )

    # --------------------------------------------------
    # TEST 9: Check page numbers.
    # --------------------------------------------------

    def test_page_numbers(self):

        document = self.load_sample()

        for element in document.elements:

            if element.page_no is not None:
                self.assertGreaterEqual(element.page_no, 1)

    # --------------------------------------------------
    # TEST 10: Check bounding boxes.
    # --------------------------------------------------

    def test_bounding_boxes(self):

        document = self.load_sample()

        for element in document.elements:

            if element.bbox is not None:

                self.assertGreater(
                    element.bbox.x1,
                    element.bbox.x0
                )

                self.assertGreater(
                    element.bbox.y1,
                    element.bbox.y0
                )

    # --------------------------------------------------
    # TEST 11: Check document metadata.
    # --------------------------------------------------

    def test_document_metadata(self):

        document = self.load_sample()

        self.assertEqual(
            document.metadata.get("parser"),
            "Docling"
        )

        self.assertEqual(
            document.metadata.get("total_elements"),
            len(document.elements)
        )

    # --------------------------------------------------
    # TEST 12: Check missing PDF handling.
    # --------------------------------------------------

    def test_missing_pdf(self):

        with self.assertRaises(FileNotFoundError):

            self.loader.load_pdf(
                "data/raw/missing_file.pdf"
            )

    # --------------------------------------------------
    # TEST 13: Check invalid extension handling.
    # --------------------------------------------------

    def test_invalid_file_extension(self):

        with self.assertRaises(ValueError):

            self.loader.load_pdf(
                "data/raw/sample.txt"
            )


# ==================================================
# RUN TESTS
# ==================================================

if __name__ == "__main__":

    unittest.main(verbosity=2)
