
import os
import mimetypes
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from PIL import Image, ImageOps, UnidentifiedImageError
from rapidocr import RapidOCR
from google import genai
from google.genai import types


class VisionProcessor:

    SUPPORTED_EXTENSIONS = {
        ".png", ".jpg", ".jpeg",
        ".bmp", ".tiff", ".webp",
    }

    def __init__(
        self,
        enable_ocr: bool = True,
        enable_gemini: bool = True,
        gemini_model: str = "gemini-3.5-flash-lite",
    ):
        self.enable_ocr = enable_ocr
        self.enable_gemini = enable_gemini
        self.gemini_model = gemini_model

        self._ocr_engine = None
        self._gemini_client = None

    def validate_image(self, image_path) -> Path:

        path = Path(image_path).expanduser()

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}"
            )

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        if not path.is_file():
            raise ValueError("Image path must point to a file.")

        if path.stat().st_size == 0:
            raise ValueError("Image file is empty.")

        return path.resolve()

    def load_image(self, image_path: Path):

        try:
            with Image.open(image_path) as pil_image:

                original_format = pil_image.format
                original_mode = pil_image.mode

                corrected_image = ImageOps.exif_transpose(
                    pil_image
                ).convert("RGB")

                corrected_image.load()

                width, height = corrected_image.size

                if width <= 0 or height <= 0:
                    raise ValueError("Invalid image dimensions.")

                image_rgb = np.array(corrected_image)

        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(
                f"Unable to decode image: {image_path}"
            ) from exc

        return (
            image_rgb,
            width,
            height,
            original_format,
            original_mode,
        )

    def extract_text(self, image_rgb) -> dict[str, Any]:

        try:
            if self._ocr_engine is None:
                self._ocr_engine = RapidOCR()

            result = self._ocr_engine(image_rgb)

            texts = getattr(result, "txts", None) or []
            scores = getattr(result, "scores", None) or []

            extracted = []

            for index, text in enumerate(texts):

                if not text or not str(text).strip():
                    continue

                score = (
                    float(scores[index])
                    if index < len(scores)
                    else None
                )

                extracted.append({
                    "text": str(text).strip(),
                    "confidence": score,
                })

            full_text = "\n".join(
                item["text"] for item in extracted
            )

            return {
                "ocr_text": full_text or None,
                "ocr_results": extracted,
                "ocr_status": (
                    "SUCCESS"
                    if extracted
                    else "NO_TEXT_DETECTED"
                ),
                "ocr_engine": "RapidOCR",
            }

        except Exception as error:

            return {
                "ocr_text": None,
                "ocr_results": [],
                "ocr_status": "ERROR",
                "ocr_engine": "RapidOCR",
                "ocr_error": str(error),
            }

    def analyze_with_gemini(
        self,
        image_path: Path,
    ) -> dict[str, Any]:

        if not self.enable_gemini:
            return {
                "status": "DISABLED",
                "model": self.gemini_model,
            }

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return {
                "status": "NOT_CONFIGURED",
                "model": self.gemini_model,
                "message": "GEMINI_API_KEY is not set.",
            }

        try:
            if self._gemini_client is None:
                self._gemini_client = genai.Client(
                    api_key=api_key
                )

            mime_type, _ = mimetypes.guess_type(
                str(image_path)
            )

            if not mime_type or not mime_type.startswith("image/"):
                mime_type = "image/jpeg"

            image_path = Path(image_path)    
            image_bytes = image_path.read_bytes()

            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            )

            prompt = """
Analyze this image for a multimodal educational
document ingestion system.

Identify:
1. A concise description of the image.
2. Main objects, entities, and visual elements.
3. Whether it contains a diagram, chart, graph,
   flowchart, table, or educational illustration.
4. The relationships and connections between
   visual elements, if present.
5. Important labels or visible text.
6. Relevant educational meaning.

Do not invent unreadable text or uncertain details.
Clearly distinguish observations from assumptions.
Return a concise, structured explanation.
"""

            response = self._gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=[
                    prompt,
                    image_part,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1200,
                ),
            )

            description = response.text

            if not description:
                return {
                    "status": "EMPTY_RESPONSE",
                    "model": self.gemini_model,
                }

            return {
                "status": "SUCCESS",
                "model": self.gemini_model,
                "description": description.strip(),
            }

        except Exception as error:

            return {
                "status": "ERROR",
                "model": self.gemini_model,
                "error": str(error),
            }

    def process_image(self, image_path) -> dict[str, Any]:

        path = self.validate_image(image_path)

        (
            image_rgb,
            width,
            height,
            image_format,
            image_mode,
        ) = self.load_image(path)

        image_cv = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR,
        )

        gray = cv2.cvtColor(
            image_cv,
            cv2.COLOR_BGR2GRAY,
        )

        edges = cv2.Canny(
            gray,
            100,
            200,
        )

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        metadata = {
            "processor": "OpenCV",
            "image_name": path.name,
            "image_path": str(path),
            "width": width,
            "height": height,
            "format": image_format,
            "mode": image_mode,
            "file_size": path.stat().st_size,
            "aspect_ratio": round(width / height, 4),
            "edge_pixels": int(cv2.countNonZero(edges)),
            "detected_contours": len(contours),
            "processing_status": "SUCCESS",
        }

        if self.enable_ocr:
            metadata.update(
                self.extract_text(image_rgb)
            )
        else:
            metadata["ocr_text"] = None
            metadata["ocr_status"] = "DISABLED"

        metadata["gemini_vision"] = (
            self.analyze_with_gemini(path)
        )

        return metadata

    def process_element(self, element):

        asset_path = getattr(
            element,
            "asset_path",
            None,
        )

        if not asset_path:
            return element

        vision_metadata = self.process_image(asset_path)

        existing_metadata = (
            getattr(element, "metadata", {}) or {}
        )

        updated_metadata = {
            **existing_metadata,
            "vision": vision_metadata,
        }

        return element.model_copy(
            update={"metadata": updated_metadata}
        )
