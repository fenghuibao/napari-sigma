from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
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
        with tempfile.TemporaryDirectory() as tmp:
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
QApplication.processEvents()
assert 'matplotlib' not in sys.modules, 'Matplotlib eagerly imported'
assert 'openpyxl' not in sys.modules, 'openpyxl eagerly imported'
assert 'torch' not in sys.modules, 'Torch eagerly imported'
assert w._analysis_plot_axes is None
w._panel_tabs.setCurrentWidget(w._analysis_tab)
QApplication.processEvents()
assert w._analysis_plot_axes is not None, 'Plot not initialized on first use'
w.dispose()
w.close()
v.close()
'''
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, MPLCONFIGDIR=tmp)
            result = subprocess.run([sys.executable, "-c", code], env=env,
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

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
