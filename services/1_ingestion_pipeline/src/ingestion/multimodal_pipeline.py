
# ==========================================================
# CogitoLearn - Multimodal Ingestion Pipeline
# Stage 1: Standalone Image Ingestion
# SHA-256 + Vision Integration
# ==========================================================

import json
from pathlib import Path
from typing import Union, Iterable

from src.ingestion.schemas import (
    UnifiedDocumentAST,
    RawMediaStream,
    IngestedElement,
    MediaModality,
    ContentType,
)

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.audio_loader import AudioLoader
from src.ingestion.video_loader import VideoLoader
from src.ingestion.vision_processor import VisionProcessor
from src.ingestion.fingerprinting import FingerprintingEngine


class MultimodalPipeline:

    DOCUMENT_FORMATS = {
        ".pdf", ".docx", ".pptx"
    }

    AUDIO_FORMATS = {
        ".wav", ".mp3", ".m4a", ".ogg", ".flac"
    }

    VIDEO_FORMATS = {
        ".mp4", ".mov", ".mkv", ".avi",
        ".webm", ".mpeg", ".mpg", ".m4v"
    }

    IMAGE_FORMATS = {
        ".png", ".jpg", ".jpeg",
        ".webp", ".bmp", ".tif", ".tiff"
    }

    # ------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------

    def __init__(
        self,
        model_name="small",
        device="cpu",
        compute_type="int8",
        frame_interval_seconds=10,
        enable_ocr=True,
    ):

        self.document_loader = DocumentLoader()

        self.audio_loader = AudioLoader(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
        )

        self.video_loader = VideoLoader(
            frame_interval_seconds=frame_interval_seconds,
            audio_loader=self.audio_loader,
        )

        self.vision_processor = VisionProcessor(
            enable_ocr=enable_ocr
        )

        # SHA-256 Fingerprinting Engine
        self.fingerprinting_engine = FingerprintingEngine()

    # ------------------------------------------------------
    # 1. INPUT VALIDATION
    # ------------------------------------------------------

    def validate_input(
        self,
        file_path: Union[str, Path]
    ) -> Path:

        path = Path(file_path).expanduser()

        if not path.exists():
            raise FileNotFoundError(
                f"Input file does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                "Input must be a file."
            )

        path = path.resolve()

        supported = (
            self.DOCUMENT_FORMATS
            | self.AUDIO_FORMATS
            | self.VIDEO_FORMATS
            | self.IMAGE_FORMATS
        )

        if path.suffix.lower() not in supported:
            raise ValueError(
                f"Unsupported input format: {path.suffix}"
            )

        if path.stat().st_size == 0:
            raise ValueError(
                "Input file is empty."
            )

        return path

    # ------------------------------------------------------
    # 2. STANDALONE IMAGE LOADER
    # ------------------------------------------------------

    def _load_standalone_image(
        self,
        path: Path
    ) -> UnifiedDocumentAST:

        (
            image_rgb,
            width,
            height,
            image_format,
            image_mode,
        ) = self.vision_processor.load_image(path)

        file_size = path.stat().st_size

        source = RawMediaStream(
            file_name=path.name,
            file_path=str(path),
            modality=MediaModality.IMAGE,
            file_size=file_size,
            metadata={
                "source_type": "STANDALONE_IMAGE",
                "image_format": image_format,
                "width": width,
                "height": height,
                "mode": image_mode,
            },
        )

        element = IngestedElement(
            content=f"Standalone image: {path.name}",
            content_type=ContentType.IMAGE,
            asset_path=str(path),
            metadata={
                "source_engine": "CogitoLearn Vision",
                "standalone_image": True,
                "image_format": image_format,
                "width": width,
                "height": height,
                "mode": image_mode,
            },
        )

        return UnifiedDocumentAST(
            source=source,
            elements=[element],
            metadata={
                "pipeline": "CogitoLearn Multimodal Pipeline",
                "ingestion_type": "STANDALONE_IMAGE",
                "image_processing_status": "PENDING",
            },
        )

    # ------------------------------------------------------
    # 3. MODALITY ROUTING
    # ------------------------------------------------------

    def _load_by_modality(
        self,
        path: Path
    ) -> UnifiedDocumentAST:

        extension = path.suffix.lower()

        if extension in self.DOCUMENT_FORMATS:
            return self.document_loader.load_document(
                str(path)
            )

        if extension in self.AUDIO_FORMATS:
            return self.audio_loader.load_audio(
                str(path)
            )

        if extension in self.VIDEO_FORMATS:
            return self.video_loader.load_video(
                str(path)
            )

        if extension in self.IMAGE_FORMATS:
            return self._load_standalone_image(path)

        raise ValueError(
            f"Unsupported modality: {extension}"
        )

    # ------------------------------------------------------
    # 4. VISUAL ELEMENT PROCESSING
    # ------------------------------------------------------

    def _process_visual_elements(
        self,
        ast: UnifiedDocumentAST
    ) -> UnifiedDocumentAST:

        processed_elements = []
        vision_errors = []

        for element in ast.elements:

            if not element.asset_path:
                processed_elements.append(element)
                continue

            asset_path = Path(element.asset_path)

            if (
                asset_path.suffix.lower()
                not in self.vision_processor.SUPPORTED_EXTENSIONS
            ):
                processed_elements.append(element)
                continue

            try:

                processed_element = (
                    self.vision_processor.process_element(
                        element
                    )
                )

                processed_elements.append(
                    processed_element
                )

            except Exception as error:

                vision_errors.append({
                    "element_id": element.element_id,
                    "asset_path": str(asset_path),
                    "error": str(error),
                })

                # Preserve the original element.
                processed_elements.append(element)

        updated_metadata = dict(ast.metadata)

        updated_metadata["vision_processing_status"] = (
            "FAILED" if vision_errors else "SUCCESS"
        )

        if vision_errors:
            updated_metadata["vision_processing_errors"] = (
                vision_errors
            )

        if ast.source.modality == MediaModality.IMAGE:

            updated_metadata["image_processing_status"] = (
                "FAILED" if vision_errors else "SUCCESS"
            )

        return ast.model_copy(
            update={
                "elements": processed_elements,
                "metadata": updated_metadata,
            }
        )

    # ------------------------------------------------------
    # 5. SHA-256 FINGERPRINT INTEGRATION
    # ------------------------------------------------------

    def _attach_fingerprint(
        self,
        ast: UnifiedDocumentAST,
        path: Path
    ) -> UnifiedDocumentAST:

        sha256 = (
            self.fingerprinting_engine.calculate_sha256(
                str(path)
            )
        )

        updated_source = ast.source.model_copy(
            update={
                "sha256": sha256
            }
        )

        updated_metadata = dict(ast.metadata)

        updated_metadata["fingerprinting"] = {
            "algorithm": "SHA-256",
            "status": "SUCCESS",
        }

        return ast.model_copy(
            update={
                "source": updated_source,
                "metadata": updated_metadata,
            }
        )

    # ------------------------------------------------------
    # 6. MAIN INGESTION
    # ------------------------------------------------------

    def ingest(
        self,
        file_path: Union[str, Path]
    ) -> UnifiedDocumentAST:

        path = self.validate_input(file_path)

        print(f"\nProcessing: {path.name}")

        # Load according to modality.
        ast = self._load_by_modality(path)

        # Attach SHA-256 fingerprint.
        ast = self._attach_fingerprint(
            ast,
            path
        )

        # Process image assets and diagrams.
        final_ast = self._process_visual_elements(
            ast
        )

        print("Ingestion completed.")
        print("Modality:", final_ast.source.modality)
        print("Elements:", len(final_ast.elements))
        print("SHA-256:", final_ast.source.sha256)

        return final_ast

    # ------------------------------------------------------
    # 7. BATCH INGESTION
    # ------------------------------------------------------

    def ingest_many(
        self,
        file_paths: Iterable[Union[str, Path]]
    ) -> list:

        results = []

        for file_path in file_paths:

            try:

                ast = self.ingest(file_path)

                results.append({
                    "status": "SUCCESS",
                    "file": str(file_path),
                    "document": ast,
                    "error": None,
                })

            except Exception as error:

                results.append({
                    "status": "FAILED",
                    "file": str(file_path),
                    "document": None,
                    "error": str(error),
                })

        return results

    # ------------------------------------------------------
    # 8. JSON SERIALIZATION
    # ------------------------------------------------------

    @staticmethod
    def save_ast(
        ast: UnifiedDocumentAST,
        output_path: Union[str, Path]
    ) -> Path:

        path = Path(output_path).expanduser()

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        path.write_text(
            ast.model_dump_json(indent=4),
            encoding="utf-8"
        )

        return path

    # ------------------------------------------------------
    # 9. SUMMARY
    # ------------------------------------------------------

    @staticmethod
    def get_summary(
        ast: UnifiedDocumentAST
    ) -> dict:

        counts = {}

        for element in ast.elements:

            content_type = str(
                element.content_type
            ).split(".")[-1]

            counts[content_type] = (
                counts.get(content_type, 0) + 1
            )

        return {
            "file": ast.source.file_name,
            "modality": str(
                ast.source.modality
            ).split(".")[-1],
            "file_size": ast.source.file_size,
            "sha256": ast.source.sha256,
            "total_elements": len(ast.elements),
            "element_counts": counts,
            "vision_processing_status": ast.metadata.get(
                "vision_processing_status"
            ),
            "image_processing_status": ast.metadata.get(
                "image_processing_status"
            ),
            "vision_processing_errors": ast.metadata.get(
                "vision_processing_errors", []
            ),
            "fingerprinting": ast.metadata.get(
                "fingerprinting"
            ),
        }


# ==========================================================
# 10. STANDALONE EXECUTION
# ==========================================================

if __name__ == "__main__":

    pipeline = MultimodalPipeline()

    input_dir = Path("data/raw")
    output_dir = Path("data/processed/ingestion")

    if not input_dir.exists():

        print(
            "Input directory does not exist:",
            input_dir
        )

        raise SystemExit(0)

    supported = (
        MultimodalPipeline.DOCUMENT_FORMATS
        | MultimodalPipeline.AUDIO_FORMATS
        | MultimodalPipeline.VIDEO_FORMATS
        | MultimodalPipeline.IMAGE_FORMATS
    )

    files = [
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in supported
    ]

    if not files:

        print("No supported input files found.")
        raise SystemExit(0)

    results = pipeline.ingest_many(files)

    successful = 0
    failed = 0

    for result in results:

        print("\nFile:", result["file"])
        print("Status:", result["status"])

        if result["status"] == "SUCCESS":

            successful += 1

            ast = result["document"]

            output_path = (
                output_dir / f"{ast.document_id}.json"
            )

            pipeline.save_ast(
                ast,
                output_path
            )

            print(
                json.dumps(
                    pipeline.get_summary(ast),
                    indent=4,
                    ensure_ascii=False,
                )
            )

            print("Saved:", output_path)

        else:

            failed += 1

            print("Error:", result["error"])

    print("\n========== FINAL RESULTS ==========")
    print("Total:", len(results))
    print("Successful:", successful)
    print("Failed:", failed)
