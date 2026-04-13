"""Unit tests for FeatherPGS debug capture configuration and artifact writing."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton._src.solvers.feather_pgs.debug_capture import FeatherPGSCaptureConfig


class TestFeatherPGSCapture(unittest.TestCase):
    def test_from_env_disabled_without_directory(self):
        original = {key: os.environ.get(key) for key in self._capture_env_keys()}
        try:
            for key in self._capture_env_keys():
                os.environ.pop(key, None)
            self.assertIsNone(FeatherPGSCaptureConfig.from_env())
        finally:
            self._restore_env(original)

    def test_from_env_parses_threshold_and_max_frames(self):
        original = {key: os.environ.get(key) for key in self._capture_env_keys()}
        try:
            os.environ["NEWTON_FEATHER_PGS_CAPTURE_DIR"] = "/tmp/fpgs-capture"
            os.environ["NEWTON_FEATHER_PGS_CAPTURE_VELOCITY_THRESHOLD"] = "42.5"
            os.environ["NEWTON_FEATHER_PGS_CAPTURE_MAX_FRAMES"] = "3"

            config = FeatherPGSCaptureConfig.from_env()

            self.assertIsNotNone(config)
            assert config is not None
            self.assertEqual(config.directory, Path("/tmp/fpgs-capture"))
            self.assertEqual(config.velocity_threshold, 42.5)
            self.assertEqual(config.max_frames, 3)
            self.assertTrue(config.should_capture(50.0))
            self.assertFalse(config.should_capture(10.0))
        finally:
            self._restore_env(original)

    def test_from_env_rejects_non_positive_max_frames(self):
        original = {key: os.environ.get(key) for key in self._capture_env_keys()}
        try:
            os.environ["NEWTON_FEATHER_PGS_CAPTURE_DIR"] = "/tmp/fpgs-capture"
            os.environ["NEWTON_FEATHER_PGS_CAPTURE_MAX_FRAMES"] = "0"

            with self.assertRaisesRegex(ValueError, "MAX_FRAMES"):
                FeatherPGSCaptureConfig.from_env()
        finally:
            self._restore_env(original)

    def test_write_capture_creates_json_and_npz(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = FeatherPGSCaptureConfig(directory=Path(tmp_dir), velocity_threshold=None, max_frames=1)
            metadata = {
                "step": 7,
                "capture_sequence": np.int32(0),
                "velocity_threshold": None,
            }
            arrays = {
                "rhs": np.array([[1.0, 2.0]], dtype=np.float32),
                "constraint_count": np.array([2], dtype=np.int32),
            }

            metadata_path, payload_path = config.write_capture(step=7, sequence=0, metadata=metadata, arrays=arrays)

            self.assertTrue(metadata_path.exists())
            self.assertTrue(payload_path.exists())
            self.assertEqual(metadata_path.name, "fpgs_capture_step000007_frame000.json")
            self.assertEqual(payload_path.name, "fpgs_capture_step000007_frame000.npz")

            written_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(written_metadata["step"], 7)
            self.assertEqual(written_metadata["capture_sequence"], 0)
            written_arrays = np.load(payload_path)
            np.testing.assert_array_equal(written_arrays["rhs"], arrays["rhs"])
            np.testing.assert_array_equal(written_arrays["constraint_count"], arrays["constraint_count"])

    @staticmethod
    def _capture_env_keys() -> tuple[str, ...]:
        return (
            "NEWTON_FEATHER_PGS_CAPTURE_DIR",
            "NEWTON_FEATHER_PGS_CAPTURE_VELOCITY_THRESHOLD",
            "NEWTON_FEATHER_PGS_CAPTURE_MAX_FRAMES",
        )

    @classmethod
    def _restore_env(cls, original: dict[str, str | None]) -> None:
        for key in cls._capture_env_keys():
            value = original[key]
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
