
# Import the models we want to test.
from pydantic import ValidationError

from src.ingestion.schemas import (
    RawMediaStream,
    BoundingBox,
    IngestedElement,
    UnifiedDocumentAST
)


# ==================================================
# TEST 1: Create a valid PDF input.
# ==================================================

pdf_file = RawMediaStream(
    file_name="lecture.pdf",
    file_path="data/raw/lecture.pdf",
    modality="PDF",
    file_size=2048000
)

print("TEST 1: PDF input created successfully")


# ==================================================
# TEST 2: Create a valid bounding box.
# ==================================================

box = BoundingBox(
    x0=10,
    y0=20,
    x1=100,
    y1=150
)

print("TEST 2: Bounding box created successfully")


# ==================================================
# TEST 3: Create an extracted document element.
# ==================================================

element = IngestedElement(
    content="Introduction to Artificial Intelligence",
    content_type="PROSE",
    page_no=1,
    bbox=box
)

print("TEST 3: Document element created successfully")


# ==================================================
# TEST 4: Create the unified document.
# ==================================================

document = UnifiedDocumentAST(
    source=pdf_file,
    elements=[element]
)

print("TEST 4: UnifiedDocumentAST created successfully")


# ==================================================
# TEST 5: Check invalid input rejection.
# ==================================================

try:

    RawMediaStream(
        file_name="lecture.pdf",
        file_path="data/raw/lecture.pdf",
        modality="IMAGE"
    )

except ValidationError:

    print("TEST 5: Invalid modality rejected successfully")


# ==================================================
# TEST 6: Display the final document.
# ==================================================

print("\nFINAL DOCUMENT:")
print(document.model_dump_json(indent=4))

print("\nALL TESTS COMPLETED!")


# ==================================================
# TEST 7: Validate an audio file.
# ==================================================

audio_file = RawMediaStream(
    file_name="lecture_audio.mp3",
    file_path="data/raw/lecture_audio.mp3",
    modality="AUDIO",
    file_size=5000000
)

print("TEST 7: Audio input created successfully")


# ==================================================
# TEST 8: Validate a video file.
# ==================================================

video_file = RawMediaStream(
    file_name="lecture_video.mp4",
    file_path="data/raw/lecture_video.mp4",
    modality="VIDEO",
    file_size=15000000
)

print("TEST 8: Video input created successfully")


# ==================================================
# TEST 9: Validate an audio transcription segment.
# ==================================================

audio_element = IngestedElement(
    content="Machine learning is a branch of AI",
    content_type="ASR_SPAN",
    start_ms=1200,
    end_ms=4500
)

print("TEST 9: Audio timestamps created successfully")


# ==================================================
# TEST 10: Reject invalid timestamps.
# ==================================================

try:

    IngestedElement(
        content="Invalid audio segment",
        content_type="ASR_SPAN",
        start_ms=5000,
        end_ms=2000
    )

except ValidationError:

    print("TEST 10: Invalid timestamps rejected successfully")


print("\nALL 10 TESTS COMPLETED!")
