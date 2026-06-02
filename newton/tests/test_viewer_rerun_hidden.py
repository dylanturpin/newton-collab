# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

# ruff: noqa: PLC0415


def _install_lightweight_newton_stubs() -> None:
    """Let this mocked Rerun test run in environments without Warp."""

    if importlib.util.find_spec("warp") is not None:
        return

    newton_root = Path(__file__).resolve().parents[1]

    warp = types.ModuleType("warp")

    class _WarpArray:
        @classmethod
        def __class_getitem__(cls, _item):
            return cls

    warp.array = _WarpArray
    warp.vec2 = object
    warp.vec3 = object
    warp.vec4 = object
    warp.transform = object
    warp.float32 = float
    warp.int32 = int
    warp.uint32 = int
    sys.modules.setdefault("warp", warp)

    newton = types.ModuleType("newton")
    newton.__path__ = [str(newton_root)]
    newton.State = type("State", (), {})
    newton.Mesh = type("Mesh", (), {})
    newton.Heightfield = type("Heightfield", (), {})
    newton.GeoType = types.SimpleNamespace(PLANE=0)
    sys.modules.setdefault("newton", newton)

    for package_name, package_path in (
        ("newton._src", newton_root / "_src"),
        ("newton._src.viewer", newton_root / "_src" / "viewer"),
        ("newton._src.core", newton_root / "_src" / "core"),
        ("newton._src.utils", newton_root / "_src" / "utils"),
    ):
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        sys.modules.setdefault(package_name, package)

    core_types = types.ModuleType("newton._src.core.types")
    core_types.override = lambda fn: fn
    sys.modules.setdefault("newton._src.core.types", core_types)

    viewer = types.ModuleType("newton._src.viewer.viewer")

    class ViewerBase:
        def __init__(self):
            self.device = "cpu"
            self.time = 0.0

    viewer.ViewerBase = ViewerBase
    viewer.is_jupyter_notebook = lambda: False
    sys.modules.setdefault("newton._src.viewer.viewer", viewer)

    mesh_utils = types.ModuleType("newton._src.utils.mesh")
    mesh_utils.compute_vertex_normals = lambda points, indices, device=None: points
    sys.modules.setdefault("newton._src.utils.mesh", mesh_utils)

    texture_utils = types.ModuleType("newton._src.utils.texture")
    texture_utils.load_texture = lambda texture: texture
    texture_utils.normalize_texture = lambda texture, **_kwargs: None if texture is None else np.asarray(texture)
    sys.modules.setdefault("newton._src.utils.texture", texture_utils)


_install_lightweight_newton_stubs()


class TestViewerRerunHidden(unittest.TestCase):
    """Regression tests for the hidden parameter in ViewerRerun log_mesh/log_instances."""

    def _create_viewer(self):
        """Create a ViewerRerun with mocked rerun backend."""
        self.mock_rr = Mock()
        self.mock_rr.init = Mock()
        self.mock_rr.spawn = Mock()
        self.mock_rr.connect_grpc = Mock()
        self.mock_rr.set_time = Mock()
        self.mock_rr.save = Mock()
        self.mock_rr.log = Mock()
        self.mock_rr.Clear = Mock(return_value=Mock())
        self.mock_rr.Mesh3D = Mock(return_value=Mock())
        self.mock_rr.InstancePoses3D = Mock(return_value=Mock())
        self.mock_rr.LineStrips3D = Mock(return_value=Mock())
        self.mock_rr.Points3D = Mock(return_value=Mock())

        self.mock_rrb = Mock()
        self.mock_rrb.Blueprint = Mock(return_value=Mock())
        self.mock_rrb.Horizontal = Mock(return_value=Mock())
        self.mock_rrb.Spatial3DView = Mock(return_value=Mock())
        self.mock_rrb.TimePanel = Mock(return_value=Mock())
        self.mock_rrb.TimeSeriesView = Mock(return_value=Mock())

        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        viewer = ViewerRerun(serve_web_viewer=False)

        return viewer

    def _make_mock_wp_array(self, data):
        """Create a mock warp array that behaves enough for ViewerRerun."""
        arr = Mock()
        np_data = np.array(data, dtype=np.float32)
        arr.numpy.return_value = np_data
        arr.dtype = Mock()
        arr.device = "cpu"
        arr.shape = np_data.shape
        arr.__len__ = lambda self_: len(np_data)
        return arr

    def test_log_mesh_hidden_skips_log(self):
        """log_mesh(hidden=True) should store the mesh in _meshes but not render them."""
        viewer = self._create_viewer()

        points = self._make_mock_wp_array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        indices = self._make_mock_wp_array([0, 1, 2])

        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            viewer.log_mesh("hidden_mesh", points, indices, hidden=True)

        self.assertIn("hidden_mesh", viewer._meshes)
        self.mock_rr.log.assert_not_called()

    def test_log_mesh_hidden_preserves_uvs_and_texture(self):
        """Hidden mesh templates should retain shading data for later instancing."""
        viewer = self._create_viewer()

        points = self._make_mock_wp_array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        indices = self._make_mock_wp_array([0, 1, 2])
        normals = self._make_mock_wp_array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        uvs = self._make_mock_wp_array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        texture = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 255]],
            ],
            dtype=np.uint8,
        )

        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            viewer.log_mesh(
                "hidden_mesh_textured", points, indices, normals=normals, uvs=uvs, texture=texture, hidden=True
            )

        mesh_data = viewer._meshes["hidden_mesh_textured"]
        self.assertIsNotNone(mesh_data["normals"])
        self.assertIsNotNone(mesh_data["uvs"])
        self.assertIsNotNone(mesh_data["texture_image"])
        np.testing.assert_allclose(mesh_data["uvs"][:, 1], np.array([0.8, 0.6, 0.4], dtype=np.float32))
        self.mock_rr.log.assert_not_called()

    def test_log_instances_hidden_clears_entity(self):
        """log_instances(hidden=True) should clear a previously visible entity."""
        viewer = self._create_viewer()

        # Manually register a mesh and instance so log_instances sees them
        viewer._meshes["my_mesh"] = {
            "points": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            "indices": np.array([[0, 1, 2]], dtype=np.uint32),
            "normals": np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32),
            "uvs": None,
            "texture_image": None,
            "texture_buffer": None,
            "texture_format": None,
        }
        viewer._instances["my_instance"] = Mock()

        xforms = self._make_mock_wp_array([[0, 0, 0, 0, 0, 0, 1]])
        scales = self._make_mock_wp_array([[1, 1, 1]])

        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            viewer.log_instances("my_instance", "my_mesh", xforms, scales, colors=None, materials=None, hidden=True)

        # Verify rr.Clear was constructed and logged
        self.mock_rr.Clear.assert_called_once_with(recursive=False)
        self.mock_rr.log.assert_called_once_with("my_instance", self.mock_rr.Clear.return_value)

    def test_log_instances_hidden_noop_when_not_created(self):
        """log_instances(hidden=True) for a never-visible entity should not crash or log."""
        viewer = self._create_viewer()

        # Register a mesh but do NOT create any instances
        viewer._meshes["my_mesh"] = {
            "points": np.array([[0, 0, 0]], dtype=np.float32),
            "indices": np.array([[0, 0, 0]], dtype=np.uint32),
            "normals": np.array([[0, 0, 1]], dtype=np.float32),
            "uvs": None,
            "texture_image": None,
            "texture_buffer": None,
            "texture_format": None,
        }

        xforms = self._make_mock_wp_array([[0, 0, 0, 0, 0, 0, 1]])

        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            # Reset mock to track only calls from this point
            self.mock_rr.log.reset_mock()
            viewer.log_instances(
                "new_instance", "my_mesh", xforms, scales=None, colors=None, materials=None, hidden=True
            )

        # No rr.log call should have been made
        self.mock_rr.log.assert_not_called()

    def test_log_lines_hidden_and_empty_clear_entity(self):
        """log_lines should clear stale line entities when hidden or empty."""
        viewer = self._create_viewer()

        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            viewer.log_lines("/contacts", None, None, None, hidden=True)
            viewer.log_lines("/contacts", self._make_mock_wp_array([]), self._make_mock_wp_array([]), None)

        self.assertEqual(self.mock_rr.Clear.call_count, 2)
        self.assertEqual(self.mock_rr.log.call_count, 2)
        self.mock_rr.log.assert_called_with("/contacts", self.mock_rr.Clear.return_value)

    def test_log_points_hidden_and_empty_clear_entity(self):
        """log_points should clear stale point entities when hidden or empty."""
        viewer = self._create_viewer()

        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            viewer.log_points("/contacts/points", None, hidden=True)
            viewer.log_points("/contacts/points", self._make_mock_wp_array([]))

        self.assertEqual(self.mock_rr.Clear.call_count, 2)
        self.assertEqual(self.mock_rr.log.call_count, 2)
        self.mock_rr.log.assert_called_with("/contacts/points", self.mock_rr.Clear.return_value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
