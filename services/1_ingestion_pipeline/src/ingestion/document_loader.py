
# ==================================================
# CogitoLearn - Multiformat Document Ingestion Engine
# File: document_loader.py
# Purpose:
# Extract structured content from PDF, DOCX and PPTX
# using Docling and convert it into UnifiedDocumentAST.
# ==================================================

from pathlib import Path
from typing import Optional, Any
import re

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling_core.types.doc import PictureItem

from src.ingestion.fingerprinting import FingerprintingEngine

from src.ingestion.schemas import (
    MediaModality,
    ContentType,
    RawMediaStream,
    BoundingBox,
    IngestedElement,
    UnifiedDocumentAST,
)


# ==================================================
# 1. DOCUMENT LOADER
# ==================================================

class DocumentLoader:

    SUPPORTED_FORMATS = {
        ".pdf": MediaModality.PDF,
        ".docx": MediaModality.DOCX,
        ".pptx": MediaModality.PPTX,
    }

    def __init__(
        self,
        image_output_dir: str = "data/processed/extracted_images",
    ):

        self.image_output_dir = Path(image_output_dir)

        # Configure PDF image extraction.
        pipeline_options = PdfPipelineOptions(
            generate_picture_images=True,
            images_scale=1.0,
        )

        # Docling uses its default document pipelines
        # for DOCX and PPTX.
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        self.fingerprinting_engine = FingerprintingEngine()


    # ==================================================
    # 2. VALIDATE INPUT
    # ==================================================

    def validate_file(self, file_path: str) -> Path:

        path = Path(file_path).expanduser()

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                "Unsupported format. "
                "Supported formats: PDF, DOCX, PPTX."
            )

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                "The provided path must point to a file."
            )

        return path.resolve()


    # ==================================================
    # 3. EXTRACT SOURCE FINGERPRINT
    # ==================================================

    def calculate_fingerprint(self, path: Path) -> str:

        fingerprint = (
            self.fingerprinting_engine.calculate_sha256(
                str(path)
            )
        )

        if not fingerprint:
            raise RuntimeError(
                "SHA-256 fingerprint calculation failed."
            )

        fingerprint = str(fingerprint).lower()

        if not re.fullmatch(r"[a-f0-9]{64}", fingerprint):
            raise ValueError(
                "FingerprintEngine returned an invalid SHA-256."
            )

        return fingerprint


    # ==================================================
    # 4. EXTRACT PAGE NUMBER
    # ==================================================

    def extract_page_number(
        self,
        item,
    ) -> Optional[int]:

        try:
            provenance = getattr(item, "prov", None)

            if not provenance:
                return None

            page_no = int(provenance[0].page_no)

            return page_no if page_no >= 1 else None

        except (AttributeError, TypeError, ValueError, IndexError):
            return None


    # ==================================================
    # 5. EXTRACT BOUNDING BOX
    # ==================================================

    def extract_bbox(
        self,
        item,
    ) -> Optional[BoundingBox]:

        try:
            provenance = getattr(item, "prov", None)

            if not provenance:
                return None

            bbox = provenance[0].bbox

            return BoundingBox(
                x0=float(bbox.l),
                y0=float(bbox.b),
                x1=float(bbox.r),
                y1=float(bbox.t),
            )

        except (AttributeError, TypeError, ValueError, IndexError):
            return None


    # ==================================================
    # 6. CREATE PROVENANCE ID
    # ==================================================

    def create_provenance_id(
        self,
        fingerprint: str,
        page_no: Optional[int],
        element_index: int,
    ) -> str:

        page_reference = (
            str(page_no)
            if page_no is not None
            else "unknown-page"
        )

        return (
            f"{fingerprint}:"
            f"{page_reference}:"
            f"{element_index}"
        )


    # ==================================================
    # 7. EXTRACT DOC ITEM LABEL
    # ==================================================

    @staticmethod
    def get_item_label(item) -> str:

        label = getattr(item, "label", "")

        return str(label).lower().split(".")[-1]


    # ==================================================
    # 8. EXTRACT STRUCTURED CONTENT
    # ==================================================

    def extract_content(
        self,
        item,
        document,
    ) -> tuple[str, ContentType]:

        label = self.get_item_label(item)

        # ------------------------------------------
        # Tables -> Markdown
        # ------------------------------------------

        if label == "table":

            try:
                content = item.export_to_markdown(
                    doc=document
                )

                if content and content.strip():
                    return content, ContentType.TABLE_MD

            except Exception as exc:
                raise RuntimeError(
                    "Failed to export a document table "
                    "to Markdown."
                ) from exc

            fallback = getattr(item, "text", "") or ""

            if fallback.strip():
                return fallback, ContentType.TABLE_MD

            return "", ContentType.TABLE_MD

        # ------------------------------------------
        # Mathematical formulas -> LaTeX text
        # ------------------------------------------

        if label == "formula":

            content = getattr(item, "text", "") or ""

            if not content.strip():
                return "", ContentType.LATEX_EQ

            return content, ContentType.LATEX_EQ

        # ------------------------------------------
        # Ordinary document content
        # ------------------------------------------

        content = getattr(item, "text", "") or ""

        if label in {"picture", "image"}:
            return content, ContentType.IMAGE

        return content, ContentType.PROSE


    # ==================================================
    # 9. PREPARE IMAGE OUTPUT DIRECTORY
    # ==================================================

    def get_image_directory(
        self,
        source_path: Path,
        fingerprint: str,
    ) -> Path:

        safe_name = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            source_path.stem,
        )

        directory = (
            self.image_output_dir
            / f"{safe_name}_{fingerprint[:12]}"
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory


    # ==================================================
    # 10. SAVE EXTRACTED PICTURE
    # ==================================================

    def save_picture(
        self,
        item,
        document,
        output_dir: Path,
        picture_index: int,
    ) -> Optional[Path]:

        image = item.get_image(document)

        if image is None:
            return None

        image_path = (
            output_dir
            / f"picture_{picture_index:04d}.png"
        )

        image.save(
            image_path,
            format="PNG",
        )

        return image_path.resolve()


    # ==================================================
    # 11. BUILD ELEMENT METADATA
    # ==================================================

    def build_element_metadata(
        self,
        item,
        item_index: int,
        level: Any,
        bbox: Optional[BoundingBox],
        asset_path: Optional[Path] = None,
    ) -> dict:

        metadata = {
            "source_engine": "Docling",
            "item_index": item_index,
            "docling_label": self.get_item_label(item),
            "hierarchy_level": str(level),
        }

        if bbox is not None:
            metadata["bbox"] = bbox.model_dump()

        if asset_path is not None:
            metadata["asset_format"] = asset_path.suffix.lower()
            metadata["asset_name"] = asset_path.name

        return metadata


    # ==================================================
    # 12. EXTRACT DOCUMENT ELEMENTS
    # ==================================================

    def extract_elements(
        self,
        document,
        source_path: Path,
        fingerprint: str,
    ) -> list[IngestedElement]:

        elements = []

        image_directory = self.get_image_directory(
            source_path,
            fingerprint,
        )

        picture_index = 0

        for item_index, (item, level) in enumerate(
            document.iterate_items()
        ):

            label = self.get_item_label(item)

            page_no = self.extract_page_number(item)

            bbox = self.extract_bbox(item)

            asset_path = None

            # ------------------------------------------
            # Picture extraction
            # ------------------------------------------

            if isinstance(item, PictureItem):

                picture_index += 1

                saved_image = self.save_picture(
                    item=item,
                    document=document,
                    output_dir=image_directory,
                    picture_index=picture_index,
                )

                if saved_image is None:
                    raise RuntimeError(
                        f"Failed to extract picture "
                        f"number {picture_index}."
                    )

                asset_path = saved_image

                content = (
                    f"Extracted image: {saved_image.name}"
                )

                content_type = ContentType.IMAGE

            else:

                content, content_type = self.extract_content(
                    item,
                    document,
                )

                if not content or not content.strip():
                    continue

            # ------------------------------------------
            # Provenance
            # ------------------------------------------

            provenance_id = self.create_provenance_id(
                fingerprint=fingerprint,
                page_no=page_no,
                element_index=item_index,
            )

            # ------------------------------------------
            # Element metadata
            # ------------------------------------------

            metadata = self.build_element_metadata(
                item=item,
                item_index=item_index,
                level=level,
                bbox=bbox,
                asset_path=asset_path,
            )

            # ------------------------------------------
            # Unified element
            # ------------------------------------------

            element = IngestedElement(
                provenance_id=provenance_id,
                content=content,
                content_type=content_type,
                page_no=page_no,
                bbox=bbox,
                asset_path=(
                    asset_path.as_posix()
                    if asset_path is not None
                    else None
                ),
                metadata=metadata,
            )

            elements.append(element)

        return elements


    # ==================================================
    # 13. LOAD DOCUMENT
    # ==================================================

    def load_document(
        self,
        file_path: str,
    ) -> UnifiedDocumentAST:

        # 1. Validate input.
        path = self.validate_file(file_path)

        # 2. Determine modality.
        modality = self.SUPPORTED_FORMATS[
            path.suffix.lower()
        ]

        # 3. Calculate SHA-256.
        fingerprint = self.calculate_fingerprint(path)

        # 4. Read file metadata.
        file_size = path.stat().st_size

        if file_size == 0:
            raise ValueError(
                "Cannot ingest an empty document."
            )

        # 5. Create source contract.
        source = RawMediaStream(
            file_name=path.name,
            file_path=str(path),
            modality=modality,
            file_size=file_size,
            sha256=fingerprint,
            metadata={
                "extension": path.suffix.lower(),
            },
        )

        # 6. Convert document using Docling.
        try:
            result = self.converter.convert(
                str(path)
            )

            document = result.document

        except Exception as exc:
            raise RuntimeError(
                f"Docling failed to convert: {path.name}"
            ) from exc

        # 7. Extract document elements.
        elements = self.extract_elements(
            document=document,
            source_path=path,
            fingerprint=fingerprint,
        )

        # 8. Build unified document metadata.
        document_metadata = {
            "parser": "Docling",
            "source_format": path.suffix.lower(),
            "fingerprint_algorithm": "SHA-256",
            "total_elements": len(elements),
            "total_images": sum(
                element.content_type == ContentType.IMAGE
                for element in elements
            ),
            "total_tables": sum(
                element.content_type == ContentType.TABLE_MD
                for element in elements
            ),
            "total_equations": sum(
                element.content_type == ContentType.LATEX_EQ
                for element in elements
            ),
        }

        # 9. Create UnifiedDocumentAST.
        unified_document = UnifiedDocumentAST(
            source=source,
            elements=elements,
            metadata=document_metadata,
        )

        return unified_document


    # ==================================================
    # 14. FORMAT-SPECIFIC METHODS
    # ==================================================

    def load_pdf(self, file_path: str) -> UnifiedDocumentAST:

        if Path(file_path).suffix.lower() != ".pdf":
            raise ValueError("Expected a PDF file.")

        return self.load_document(file_path)


    def load_docx(self, file_path: str) -> UnifiedDocumentAST:

        if Path(file_path).suffix.lower() != ".docx":
            raise ValueError("Expected a DOCX file.")

        return self.load_document(file_path)


    def load_pptx(self, file_path: str) -> UnifiedDocumentAST:

        if Path(file_path).suffix.lower() != ".pptx":
            raise ValueError("Expected a PPTX file.")

        return self.load_document(file_path)


# ==================================================
# 15. BACKWARD COMPATIBILITY
# ==================================================

PDFDocumentLoader = DocumentLoader


def load_document(
    file_path: str,
) -> UnifiedDocumentAST:

    loader = DocumentLoader()

    return loader.load_document(file_path)


# ==================================================
# 16. BASIC EXECUTION
# ==================================================

if __name__ == "__main__":

    file_path = "data/raw/sample_lecture.pdf"

    loader = DocumentLoader()

    document = loader.load_document(file_path)

    print("\n========== DOCUMENT INGESTION RESULTS ==========")

    print("File:", document.source.file_name)
    print("Modality:", document.source.modality)
    print("File size:", document.source.file_size)
    print("SHA-256:", document.source.sha256)

    print("Document ID:", document.document_id)

    print("Total elements:", len(document.elements))

    print("Document metadata:", document.metadata)

    for index, element in enumerate(
        document.elements[:5],
        start=1,
    ):

        print(f"\n--- Element {index} ---")

        print("Element ID:", element.element_id)
        print("Provenance ID:", element.provenance_id)
        print("Type:", element.content_type)
        print("Page:", element.page_no)
        print("Content:", element.content[:300])
        print("Bounding Box:", element.bbox)
        print("Asset Path:", element.asset_path)

    output_path = (
        Path("data/processed")
        / f"{Path(file_path).stem}.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        document.model_dump_json(indent=4),
        encoding="utf-8",
    )

    print("\nJSON saved to:", output_path)

    print("\nDOCUMENT INGESTION COMPLETED!")
