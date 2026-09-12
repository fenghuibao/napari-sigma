from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from _fixtures import temporary_directory
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np


class GuiRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "sigma-test-numba"))
        from napari_sigma._launcher import _configure_pyqt6
        _configure_pyqt6()
        import napari
        from napari_sigma._widget import SIGMAWidget
        from qtpy.QtWidgets import QApplication
        cls.napari, cls.Widget, cls.App = napari, SIGMAWidget, QApplication

    def test_existing_edits_preserved_and_hooks_removed(self):
        from napari_sigma._writer import write_single_image
        with temporary_directory(self) as tmp:
            source = str(Path(tmp) / "source.tif")
            write_single_image(source, np.arange(256, dtype=np.float32).reshape(16, 16),
                               {"metadata": {"dims": "YX"}})
            viewer = self.napari.Viewer(show=False)
            try:
                data = np.full((16, 16), 17, np.float32)
                original_layer = viewer.add_image(data, metadata={"source": source})
                qt = viewer.window._qt_viewer
                calls = []
                def original_open(*args, **kwargs):
                    calls.append(kwargs)
                    return []
                with patch.object(qt, "_qt_open", original_open):
                    for _ in range(2):
                        panel = self.Widget(viewer)
                        viewer.window.add_dock_widget(panel, name="SIGMA regression")
                        self.App.processEvents()
                        self.assertIn(original_layer, viewer.layers)
                        self.assertIs(original_layer.data, data)
                        np.testing.assert_array_equal(original_layer.data, 17)
                        self.assertIs(qt._qt_open, original_open)
                        qt._qt_open([source], plugin="chosen-reader", choose_plugin=True, layer_type="labels")
                        self.assertEqual(calls[-1]["plugin"], "chosen-reader")
                        viewer.window.remove_dock_widget(panel)
                        self.App.processEvents()
                        self.assertTrue(panel._disposed)
                        self.assertFalse(panel._viewer_connections)
                        self.assertFalse(any(getattr(cb, "__self__", None) is panel
                                             for cb in viewer.mouse_double_click_callbacks))
                        panel.dispose()
                        panel.deleteLater()
            finally:
                viewer.close()

    def test_geometry_through_analysis_entry_point(self):
        from napari_sigma._analysis import analyze_binary_components, _major_axis_length_in_units, _perimeter_in_units
        from skimage.measure import regionprops
        hollow = np.ones((3, 3, 3), bool)
        hollow[1, 1, 1] = False
        _, rows, _, _ = analyze_binary_components(hollow, unit="um", voxel_size=(1, 1, 1))
        self.assertEqual(rows[0]["surface_area"], 60)
        mask = np.ones((5, 20), np.uint8)
        for array in (mask, mask.T):
            self.assertAlmostEqual(_major_axis_length_in_units(regionprops(array)[0], (1, 3)),
                                   regionprops(array, spacing=(1, 3))[0].axis_major_length)
        self.assertGreater(_perimeter_in_units(mask, (1, 3)), _perimeter_in_units(mask.T, (1, 3)))

    def test_unannotated_4d_processing_preserves_selected_frames(self):
        viewer = self.napari.Viewer(show=False)
        panel = self.Widget(viewer)
        try:
            data = np.arange(2 * 5 * 8 * 9, dtype=np.float32).reshape(2, 5, 8, 9)
            layer = viewer.add_image(data, rgb=False, metadata={})
            with patch.object(panel, "_selected_time_range", return_value=(0, 1)), \
                 patch.object(panel, "_selected_slice_range", return_value=(0, 4)):
                actual, dim, _, temporal = panel._extract_processing_series(layer)
            np.testing.assert_array_equal(actual, data)
            self.assertEqual(dim, 3)
            self.assertEqual(temporal, "TZYX")
        finally:
            panel.dispose()
            panel.close()
            viewer.close()

    def test_dispose_stops_superseded_analysis_workers(self):
        from napari_sigma import _widget
        from napari_sigma._analysis import AnalysisCancelledError
        viewer = self.napari.Viewer(show=False)
        panel = self.Widget(viewer)
        jobs = []
        started = [threading.Event(), threading.Event()]
        counter = [0]
        lock = threading.Lock()

        def slow_analysis(binary, **kwargs):
            with lock:
                index = counter[0]
                counter[0] += 1
            started[index].set()
            time.sleep(.1)
            if kwargs["cancel_check"]():
                raise AnalysisCancelledError("cancelled")
            return binary.astype(np.int32), [], {}, {}

        def is_running(thread):
            try:
                return thread.isRunning()
            except RuntimeError:  # already deleted after finishing
                return False

        try:
            layer = viewer.add_labels(np.ones((2, 8, 9), np.uint8), metadata={"dims": "TYX"})
            with patch.object(_widget, "analyze_binary_components", slow_analysis):
                for index in range(2):
                    panel._start_analysis_refresh(layer, frame_key=(id(layer), index))
                    jobs.append((panel._analysis_thread, panel._analysis_worker))
                    self.assertTrue(started[index].wait(3))
                panel.dispose()
                self.App.processEvents()
                self.assertFalse(any(is_running(thread) for thread, _ in jobs))
                self.assertFalse(panel._analysis_jobs)
                self.assertIsNone(panel._analysis_thread)
        finally:
            # Also safe when testing the unfixed implementation.
            for thread, worker in jobs:
                try:
                    worker.cancel()
                    worker.deleteLater()
                    thread.quit()
                    thread.wait(5000)
                except RuntimeError:
                    pass
            panel.dispose()
            panel.close()
            viewer.close()

    def test_reopened_panel_does_not_receive_old_worker_callbacks(self):
        from qtpy.QtCore import QObject, Signal
        from unittest.mock import Mock
        class Sender(QObject):
            finished = Signal()
        viewer = self.napari.Viewer(show=False)
        panel = self.Widget(viewer)
        try:
            sender = Sender()
            callback = Mock()
            panel._connect_worker_callback(sender.finished, callback)
            sender.finished.emit()  # delivery is queued to the GUI thread
            panel.dispose()
            panel._install_viewer_hooks()
            self.App.processEvents()
            callback.assert_not_called()
        finally:
            panel.dispose()
            panel.close()
            viewer.close()

    def test_completed_analysis_jobs_are_released_without_closing_panel(self):
        from napari_sigma import _widget
        viewer = self.napari.Viewer(show=False)
        panel = self.Widget(viewer)
        try:
            layer = viewer.add_labels(np.ones((2, 8, 9), np.uint8), metadata={"dims": "TYX"})
            result = (np.ones((8, 9), np.int32), [], {}, {})
            with patch.object(_widget, "analyze_binary_components", return_value=result), \
                 patch.object(panel, "_selected_analysis_layer", return_value=None):
                for index in range(3):
                    panel._start_analysis_refresh(layer, frame_key=(id(layer), index))
                deadline = time.monotonic() + 10
                while panel._analysis_jobs and time.monotonic() < deadline:
                    self.App.processEvents()
                    time.sleep(.005)
            self.assertFalse(panel._disposed)
            self.assertFalse(panel._analysis_jobs)
            self.assertIsNone(panel._analysis_thread)
        finally:
            panel.dispose()
            panel.close()
            viewer.close()

    def test_median_and_gaussian_buttons_keep_unannotated_time_frames(self):
        from napari_sigma import _widget
        viewer = self.napari.Viewer(show=False)
        panel = self.Widget(viewer)
        try:
            data = np.arange(2 * 5 * 8 * 9, dtype=np.float32).reshape(2, 5, 8, 9)
            layer = viewer.add_image(data, rgb=False, metadata={})
            panel.gaussian_enabled_checkbox.setChecked(True)
            cases = (("DenoiseWorker", "_selected_info_raw_layer", "_on_apply_denoise_clicked", "_denoise_thread"),
                     ("GaussianBackgroundWorker", "_preferred_median_input_layer", "_on_apply_gaussian_background_clicked", "_gaussian_thread"))
            for worker_name, selector, handler, thread_attr in cases:
                with self.subTest(worker=worker_name), \
                     patch.object(_widget, worker_name, wraps=getattr(_widget, worker_name)) as worker_factory, \
                     patch.object(panel, selector, return_value=layer), \
                     patch.object(panel, "_selected_time_range", return_value=(0, 1)), \
                     patch.object(panel, "_selected_slice_range", return_value=(0, 4)), \
                     patch.object(_widget.QMessageBox, "critical") as failure:
                    getattr(panel, handler)()
                    worker_factory.assert_called_once()
                    np.testing.assert_array_equal(worker_factory.call_args.args[0], data)
                    self.assertTrue(worker_factory.call_args.kwargs["temporal"])
                    deadline = time.monotonic() + 15
                    while getattr(panel, thread_attr) is not None and time.monotonic() < deadline:
                        self.App.processEvents()
                        time.sleep(.005)
                    self.assertIsNone(getattr(panel, thread_attr))
                    failure.assert_not_called()
                    self.assertEqual(panel._denoise_layer.data.shape, data.shape)
                    self.assertEqual(panel._denoise_layer.metadata["dims_out"], "TZYX")
        finally:
            panel.dispose()
            panel.close()
            viewer.close()

    def test_custom_save_retains_native_rgb_and_scale(self):
        from napari_sigma import _widget
        from napari_sigma._reader import napari_get_reader
        viewer = self.napari.Viewer(show=False)
        panel = self.Widget(viewer)
        try:
            data = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
            viewer.add_image(data, rgb=True, scale=(.3, .2))
            with temporary_directory(self) as tmp:
                path = str(Path(tmp) / "rgb.tif")
                with patch.object(_widget.QFileDialog, "getSaveFileName", return_value=(path, "")), \
                     patch.object(_widget.QMessageBox, "information"), \
                     patch.object(_widget.QMessageBox, "critical") as failure:
                    panel._save_active_layer_with_metadata()
                    failure.assert_not_called()
                layers = napari_get_reader(path=path)(path)
                self.assertEqual(len(layers), 3)
                for channel, (array, kwargs, _) in enumerate(layers):
                    np.testing.assert_array_equal(array, data[..., channel])
                    np.testing.assert_allclose(kwargs["scale"], (.3, .2))
        finally:
            panel.dispose()
            panel.close()
            viewer.close()

    def test_viewer_open_uses_registered_sigma_reader(self):
        from napari.layers import Image, Labels
        from napari_sigma._writer import write_single_image, write_single_labels
        viewer = self.napari.Viewer(show=False)
        try:
            with temporary_directory(self) as tmp:
                image = np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9)
                labels = np.full((2, 3, 8, 9), 70000, dtype=np.uint32)
                image_path = str(Path(tmp) / "image.tif")
                labels_path = str(Path(tmp) / "labels.tif")
                write_single_image(image_path, image, {"metadata": {"dims": "ZYX"}})
                write_single_labels(labels_path, labels, {"metadata": {"dims": "TZYX"}})
                opened = viewer.open([image_path, labels_path], plugin="napari-sigma", stack=False)
                self.assertEqual(len(opened), 2)
                self.assertIsInstance(opened[0], Image)
                self.assertIsInstance(opened[1], Labels)
                np.testing.assert_array_equal(opened[0].data, image)
                np.testing.assert_array_equal(opened[1].data, labels)
                self.assertEqual(opened[1].metadata["dims"], "TZYX")
        finally:
            viewer.close()

    def test_lazy_plotting_in_fresh_process(self):
        code = '''
import sys
from napari_sigma._launcher import _configure_pyqt6
_configure_pyqt6()
import napari
from napari_sigma._widget import SIGMAWidget
from qtpy.QtWidgets import QApplication
v=napari.Viewer(show=False)
w=SIGMAWidget(v)
# Construction stays lazy; the deferred device-availability check below is
# now intentionally allowed to import Torch once the panel is initialized.
assert 'torch' not in sys.modules, 'Torch imported during widget construction'
QApplication.processEvents()
assert 'matplotlib' not in sys.modules, 'Matplotlib eagerly imported'
assert 'openpyxl' not in sys.modules, 'openpyxl eagerly imported'
assert w.device_combo.findText('cpu') >= 0
assert w.device_combo.findText('auto') < 0
assert 'cv2' not in sys.modules, 'OpenCV eagerly imported'
assert w._analysis_plot_axes is None
w._panel_tabs.setCurrentWidget(w._analysis_tab)
QApplication.processEvents()
assert w._analysis_plot_axes is not None, 'Plot not initialized on first use'
w.dispose()
w.close()
v.close()
'''
        with temporary_directory(self) as tmp:
            env = dict(os.environ, MPLCONFIGDIR=tmp)
            result = subprocess.run([sys.executable, "-c", code], env=env,
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_upsampling_preserves_qt_environment(self):
        from napari_sigma._widget import _upsample_xy_bilinear
        keys = ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR", "QT_PLUGIN_PATH")
        before = {key: os.environ.get(key) for key in keys}
        frame = np.array([[0, 4], [8, 12]], dtype=np.float32)
        actual = _upsample_xy_bilinear(np.stack([frame, frame + 20]), 2)
        expected = np.array([[0, 1, 3, 4], [2, 3, 5, 6],
                             [6, 7, 9, 10], [8, 9, 11, 12]], dtype=np.float32)
        np.testing.assert_array_equal(actual, np.stack([expected, expected + 20]))
        self.assertEqual(before, {key: os.environ.get(key) for key in keys})

    def test_labels_use_same_shape_in_reader_and_widget(self):
        from napari_sigma._reader import tczyx_to_layer_data
        viewer = self.napari.Viewer(show=False)
        panel = self.Widget(viewer)
        try:
            data = np.ones((2, 1, 3, 8, 9), np.uint32) * 70000
            meta = {"layer_type": "labels", "scale_per_axis": (1, 1, 2, .3, .2)}
            expected, kwargs, _ = tczyx_to_layer_data(data, meta, "test")[0]
            layers = panel._add_tczyx_image_layers(data, meta, "test")
            self.assertEqual(layers[0].data.shape, expected.shape)
            self.assertEqual(layers[0].metadata["dims"], "TZYX")
            np.testing.assert_allclose(layers[0].scale, kwargs["scale"])
            np.testing.assert_array_equal(layers[0].data, expected)
        finally:
            panel.dispose()
            panel.close()
            viewer.close()

    def test_proximity_worker_cancellation_and_completion(self):
        from types import SimpleNamespace
        from napari_sigma._widget import ProximityWorker
        from napari_sigma._proximity import ProximityCancelledError
        data = np.zeros((8, 8), np.uint8)
        data[2:4, 2:4] = 1
        layer = SimpleNamespace(data=data, metadata={"dims": "YX"}, name="test")
        for cancelled in (False, True):
            worker = ProximityWorker([layer] * 4, dict(voxel_size=(1, 1), surface_only=False))
            results = []
            worker.finished.connect(lambda result, error: results.append((result, error)))
            if cancelled:
                worker.cancel()
            worker.run()
            self.assertEqual(len(results), 1)
            result, error = results[0]
            if cancelled:
                self.assertIsInstance(error, ProximityCancelledError)
            else:
                self.assertIsNone(error)
                self.assertEqual(result.summary_rows[0].dice, 1.)
