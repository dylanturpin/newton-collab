from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _json_ready(value: Any) -> Any:
    """Convert common NumPy and pathlib objects into JSON-friendly values."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(slots=True)
class FeatherPGSCaptureConfig:
    """Configuration for opt-in FeatherPGS debug frame capture."""

    directory: Path
    velocity_threshold: float | None
    max_frames: int

    @classmethod
    def from_env(cls) -> FeatherPGSCaptureConfig | None:
        """Load capture settings from environment variables.

        Returns:
            Parsed capture settings, or ``None`` when capture is disabled.
        """
        directory_raw = os.getenv("NEWTON_FEATHER_PGS_CAPTURE_DIR")
        if not directory_raw:
            return None

        threshold_raw = os.getenv("NEWTON_FEATHER_PGS_CAPTURE_VELOCITY_THRESHOLD")
        threshold = None if threshold_raw in (None, "") else float(threshold_raw)

        max_frames_raw = os.getenv("NEWTON_FEATHER_PGS_CAPTURE_MAX_FRAMES", "1")
        max_frames = int(max_frames_raw)
        if max_frames <= 0:
            raise ValueError("NEWTON_FEATHER_PGS_CAPTURE_MAX_FRAMES must be >= 1")

        return cls(
            directory=Path(directory_raw).expanduser(),
            velocity_threshold=threshold,
            max_frames=max_frames,
        )

    def should_capture(self, max_abs_velocity: float) -> bool:
        """Return whether the current frame should be flushed to disk."""
        if self.velocity_threshold is None:
            return True
        return max_abs_velocity >= self.velocity_threshold

    def artifact_stem(self, step: int, sequence: int) -> str:
        """Build a stable filename stem for one capture frame."""
        return f"fpgs_capture_step{step:06d}_frame{sequence:03d}"

    def write_capture(
        self,
        step: int,
        sequence: int,
        metadata: dict[str, Any],
        arrays: dict[str, np.ndarray],
    ) -> tuple[Path, Path]:
        """Write one capture artifact pair to disk.

        Args:
            step: Solver step index.
            sequence: Capture sequence number within the process lifetime.
            metadata: JSON-serializable metadata dictionary.
            arrays: NumPy arrays to store in the `.npz` payload.

        Returns:
            Tuple of `(metadata_path, payload_path)`.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = self.artifact_stem(step, sequence)
        metadata_path = self.directory / f"{stem}.json"
        payload_path = self.directory / f"{stem}.npz"

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(_json_ready(metadata), f, indent=2, sort_keys=True)
            f.write("\n")

        np.savez_compressed(payload_path, **arrays)
        return metadata_path, payload_path
