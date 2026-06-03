# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import Mock, patch

# ruff: noqa: PLC0415


class TestViewerRerunInitArgs(unittest.TestCase):
    """Unit tests for ViewerRerun initialization parameters."""

    def setUp(self):
        """Create a fresh mock rerun object for each test."""
        self.mock_rr = Mock()
        self.mock_rr.init = Mock()
        self.mock_rr.spawn = Mock()
        self.mock_rr.connect_grpc = Mock()
        self.mock_rr.set_time = Mock()
        self.mock_rr.save = Mock()

        # Mock blueprint module and components
        self.mock_rrb = Mock()
        self.mock_blueprint = Mock()
        self.mock_rrb.Blueprint = Mock(return_value=self.mock_blueprint)
        self.mock_rrb.Horizontal = Mock(return_value=Mock())
        self.mock_rrb.Spatial3DView = Mock(return_value=Mock())
        self.mock_rrb.TimePanel = Mock(return_value=Mock())
        self.mock_rrb.TimeSeriesView = Mock(return_value=Mock())

    def _mock_rr_without_grpc_server(self, *, serve_web: Mock | None = None, set_time: bool = True):
        attrs = {
            "init": Mock(),
            "spawn": Mock(),
            "connect_grpc": Mock(),
            "save": Mock(),
            "Mesh3D": Mock(),
        }
        if set_time:
            attrs["set_time"] = Mock()
        else:
            attrs["set_time_seconds"] = Mock()
        if serve_web is not None:
            attrs["serve_web"] = serve_web
        return SimpleNamespace(**attrs)

    def test_default_serves_web_viewer(self):
        """Test that ViewerRerun() with no arguments servers a web viewer."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    # Suppress deprecation warnings for cleaner test output
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _ = ViewerRerun()

                    # Verify rr.init was called with app_id as positional arg and blueprint
                    from unittest.mock import ANY

                    self.mock_rr.init.assert_called_once_with("newton-viewer", default_blueprint=ANY)

                    # Verify rr.serve_grpc() was called
                    self.mock_rr.serve_grpc.assert_called_once()
                    # Verify rr.serve_web_viewer() was called
                    self.mock_rr.serve_web_viewer.assert_called_once()

                    # Verify rr.connect_grpc() was NOT called
                    self.mock_rr.connect_grpc.assert_not_called()
                    # Verify rr.spawn() was NOT called
                    self.mock_rr.spawn.assert_not_called()

    def test_native_viewer(self):
        """Test that ViewerRerun() with no arguments spawns a viewer."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    # Suppress deprecation warnings for cleaner test output
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _ = ViewerRerun(serve_web_viewer=False)

                    # Verify rr.init was called with app_id as positional arg and blueprint
                    from unittest.mock import ANY

                    self.mock_rr.init.assert_called_once_with("newton-viewer", default_blueprint=ANY)

                    # Verify rr.spawn() was called
                    self.mock_rr.spawn.assert_called_once()

                    # Verify rr.connect_grpc() was NOT called
                    self.mock_rr.connect_grpc.assert_not_called()

    def test_custom_address_connects_grpc(self):
        """Test that ViewerRerun(address='...') connects via gRPC."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    test_address = "localhost:9876"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _ = ViewerRerun(address=test_address)

                    # Verify rr.init was called with app_id as positional arg and blueprint
                    from unittest.mock import ANY

                    self.mock_rr.init.assert_called_once_with("newton-viewer", default_blueprint=ANY)

                    # Verify rr.connect_grpc() was called with the address
                    self.mock_rr.connect_grpc.assert_called_once_with(test_address)

                    # Verify rr.spawn() was NOT called
                    self.mock_rr.spawn.assert_not_called()

    def test_custom_address_connects_grpc_in_jupyter(self):
        """Test that ViewerRerun(address='...') connects via gRPC even in Jupyter notebooks."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=True):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    test_address = "localhost:9876"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        viewer = ViewerRerun(address=test_address)

                    # Verify viewer detected Jupyter environment
                    self.assertTrue(viewer.is_jupyter_notebook)

                    # Verify rr.connect_grpc() was called with the address even in Jupyter
                    self.mock_rr.connect_grpc.assert_called_once_with(test_address)

                    # Verify rr.spawn() was NOT called
                    self.mock_rr.spawn.assert_not_called()

    def test_custom_app_id_used(self):
        """Test that custom app_id is passed to rr.init."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    custom_app_id = "my-simulation-123"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        viewer = ViewerRerun(app_id=custom_app_id)

                    # Verify rr.init was called with custom app_id as positional arg and blueprint
                    from unittest.mock import ANY

                    self.mock_rr.init.assert_called_once_with(custom_app_id, default_blueprint=ANY)

                    # Verify the viewer stored the app_id correctly
                    self.assertEqual(viewer.app_id, custom_app_id)

    def test_blueprint_passed_to_init(self):
        """Test that blueprint is created and passed to rr.init()."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _ = ViewerRerun()

                    # Verify blueprint components were created
                    self.mock_rrb.Blueprint.assert_called_once()
                    self.mock_rrb.Spatial3DView.assert_called()
                    self.mock_rrb.TimePanel.assert_called()

                    # Verify blueprint was passed to rr.init
                    call_args = self.mock_rr.init.call_args
                    self.assertIn("default_blueprint", call_args[1])
                    self.assertEqual(call_args[1]["default_blueprint"], self.mock_blueprint)

    def test_blueprint_time_panel_keeps_timeline_when_supported(self):
        """Test that older rerun TimePanel constructors still receive the timeline argument."""
        time_panel_calls = []

        def time_panel(*, timeline=None, state=None):
            time_panel_calls.append({"timeline": timeline, "state": state})
            return Mock()

        self.mock_rrb.TimePanel = time_panel

        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _ = ViewerRerun(serve_web_viewer=False)

        self.assertEqual(time_panel_calls, [{"timeline": "time", "state": "collapsed"}])

    def test_blueprint_time_panel_retries_without_timeline_when_unsupported(self):
        """Test compatibility with newer rerun TimePanel constructors that removed timeline."""
        time_panel_calls = []

        def time_panel(*, state=None):
            time_panel_calls.append({"state": state})
            return Mock()

        self.mock_rrb.TimePanel = time_panel

        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _ = ViewerRerun(serve_web_viewer=False)

        self.assertEqual(time_panel_calls, [{"state": "collapsed"}])

    def test_blueprint_time_panel_reraises_unrelated_type_error(self):
        """Test that TimePanel construction errors unrelated to timeline compatibility are not hidden."""

        def time_panel(*, timeline=None, state=None):
            raise TypeError("state must be collapsed")

        self.mock_rrb.TimePanel = time_panel

        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with self.assertRaisesRegex(TypeError, "state must be collapsed"):
                            _ = ViewerRerun(serve_web_viewer=False)

    def test_record_to_rrd_calls_save(self):
        """Test that providing record_to_rrd calls rr.save() with blueprint."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    test_path = "test_recording.rrd"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _ = ViewerRerun(record_to_rrd=test_path)

                    # Verify rr.save was called
                    self.mock_rr.save.assert_called_once()
                    call_args = self.mock_rr.save.call_args
                    self.assertEqual(call_args[0][0], test_path)
                    self.assertIn("default_blueprint", call_args[1])
                    self.assertEqual(call_args[1]["default_blueprint"], self.mock_blueprint)

    def test_record_to_rrd_skips_live_server_when_grpc_server_unavailable(self):
        """Test rerun-sdk versions without serve_grpc can still record to RRD."""
        mock_rr = self._mock_rr_without_grpc_server()

        with patch("newton._src.viewer.viewer_rerun.rr", mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        viewer = ViewerRerun(record_to_rrd="test_recording.rrd")

        mock_rr.save.assert_called_once()
        mock_rr.spawn.assert_not_called()
        mock_rr.connect_grpc.assert_not_called()
        self.assertIsNone(viewer._grpc_server_uri)

    def test_legacy_rerun_set_time_seconds_used_when_set_time_unavailable(self):
        """Test rerun-sdk 0.22 timeline setup falls back to set_time_seconds."""
        mock_rr = self._mock_rr_without_grpc_server(set_time=False)

        with patch("newton._src.viewer.viewer_rerun.rr", mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        viewer = ViewerRerun(record_to_rrd="test_recording.rrd")
                    viewer.begin_frame(1.5)

        self.assertEqual(mock_rr.set_time_seconds.call_args_list[0].args, ("time", 0.0))
        self.assertEqual(mock_rr.set_time_seconds.call_args_list[1].args, ("time", 1.5))

    def test_legacy_rerun_serve_web_used_when_grpc_server_unavailable(self):
        """Test non-recording web serving can fall back to rerun-sdk 0.22 serve_web."""
        serve_web = Mock()
        mock_rr = self._mock_rr_without_grpc_server(serve_web=serve_web)

        with patch("newton._src.viewer.viewer_rerun.rr", mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _ = ViewerRerun()

        serve_web.assert_called_once()
        call_args = serve_web.call_args
        self.assertEqual(call_args.kwargs["open_browser"], False)
        self.assertEqual(call_args.kwargs["web_port"], 9090)
        self.assertEqual(call_args.kwargs["default_blueprint"], self.mock_blueprint)
        mock_rr.save.assert_not_called()

    def test_missing_rerun_server_api_raises_without_recording(self):
        """Test that missing live server APIs still fails clearly when no RRD sink is configured."""
        mock_rr = self._mock_rr_without_grpc_server()

        with patch("newton._src.viewer.viewer_rerun.rr", mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with self.assertRaisesRegex(AttributeError, "serve_grpc"):
                            _ = ViewerRerun()

    def test_jupyter_notebook_skips_spawn(self):
        """Test that viewer is not spawned in Jupyter notebook environment."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=True):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        viewer = ViewerRerun()

                    # Verify viewer detected Jupyter environment
                    self.assertTrue(viewer.is_jupyter_notebook)

                    # Verify rr.spawn() was NOT called in Jupyter
                    self.mock_rr.spawn.assert_not_called()

                    # Verify rr.connect_grpc() was NOT called
                    self.mock_rr.connect_grpc.assert_not_called()

    def test_non_jupyter_serves_web_viewer(self):
        """Test that viewer serves web viewer in non-Jupyter environment."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        viewer = ViewerRerun()

                    # Verify viewer detected non-Jupyter environment
                    self.assertFalse(viewer.is_jupyter_notebook)

                    # Verify rr.serve_grpc() WAS called in non-Jupyter
                    self.mock_rr.serve_grpc.assert_called_once()
                    # Verify rr.serve_web_viewer() WAS called in non-Jupyter
                    self.mock_rr.serve_web_viewer.assert_called_once()

    def test_keep_historical_data_stored(self):
        """Test that keep_historical_data parameter is stored correctly."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        viewer_true = ViewerRerun(keep_historical_data=True)
                        viewer_false = ViewerRerun(keep_historical_data=False)

                    # Verify parameters were stored correctly
                    self.assertTrue(viewer_true.keep_historical_data)
                    self.assertFalse(viewer_false.keep_historical_data)

    def test_keep_scalar_history_stored(self):
        """Test that keep_scalar_history parameter is stored correctly."""
        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            with patch("newton._src.viewer.viewer_rerun.rrb", self.mock_rrb):
                with patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False):
                    from newton._src.viewer.viewer_rerun import ViewerRerun

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        viewer_true = ViewerRerun(keep_scalar_history=True)
                        viewer_false = ViewerRerun(keep_scalar_history=False)

                    # Verify parameters were stored correctly
                    self.assertTrue(viewer_true.keep_scalar_history)
                    self.assertFalse(viewer_false.keep_scalar_history)


if __name__ == "__main__":
    unittest.main(verbosity=2)
