
# ==================================================
# CogitoLearn - File Fingerprinting Engine
# File: fingerprinting.py
# Purpose:
# Generate reliable SHA-256 fingerprints
# for uploaded files using memory-efficient chunking.
# ==================================================

import hashlib
from pathlib import Path


class FingerprintingEngine:

    def __init__(
        self,
        chunk_size: int = 1024 * 1024,
    ):

        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
            raise TypeError(
                "Chunk size must be an integer."
            )

        if chunk_size <= 0:
            raise ValueError(
                "Chunk size must be positive."
            )

        self.chunk_size = chunk_size

    def calculate_sha256(
        self,
        file_path: str,
    ) -> str:

        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                "The provided path must point to a file."
            )

        if path.stat().st_size == 0:
            raise ValueError(
                "Cannot fingerprint an empty file."
            )

        sha256_hash = hashlib.sha256()

        try:

            with path.open("rb") as file:

                while True:

                    chunk = file.read(self.chunk_size)

                    if not chunk:
                        break

                    sha256_hash.update(chunk)

        except OSError as exc:

            raise RuntimeError(
                f"Failed to read file for fingerprinting: {path}"
            ) from exc

        return sha256_hash.hexdigest()
