"""Native-reader TIFF calibration must not require reloading image pixels."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

from napari_sigma._image_io import read_tiff_layer_calibration

XY = 1_000_000 / 9_087_619  # WT_1.tif's exact X/YResolution rationals.


class NativeTiffCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from napari_sigma._launcher import _configure_pyqt6
        _configure_pyqt6()
        import napari
        from napari_sigma._widget import SIGMAWidget
        from qtpy.QtWidgets import QApplication
        cls.napari, cls.Widget, cls.App = napari, SIGMAWidget, QApplication

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="sigma-calibration-")
        self.addCleanup(directory.cleanup)
        self.path = str(Path(directory.name) / "imagej.tif")
        self.shape = (2, 3, 8, 9)
        tifffile.imwrite(self.path, np.arange(np.prod(self.shape), dtype=np.uint16).reshape(self.shape),
                         imagej=True, metadata={"axes": "TZYX", "unit": "micron", "spacing": .2,
                                                "finterval": 72.579},
                         resolution=((9_087_619, 1_000_000), (9_087_619, 1_000_000)))
        self.viewer = self.napari.Viewer(show=False)
        self.addCleanup(self.viewer.close)

    def panel(self):
        panel = self.Widget(self.viewer)
        self.addCleanup(panel.close)
        self.addCleanup(panel.dispose)
        self.viewer.window.add_dock_widget(panel, name="SIGMA regression")
        return panel

    def assert_calibrated(self, layer, panel):
        np.testing.assert_allclose(layer.scale, (1, .2, XY, XY), rtol=1e-12)
        self.assertEqual(layer.metadata["dims"], "TZYX")
        self.assertEqual(layer.metadata["unit"], "um")
        self.assertEqual(layer.metadata["time_interval"], 72.579)
        panel._update_info()
        self.assertIn("vz=0.2", panel.lbl_voxpix.text())
        self.assertIn("vy=0.11004", panel.lbl_voxpix.text())
        self.assertIn("72.579", panel.lbl_time.text())

    def test_native_drop_recovers_wt1_calibration_without_reading_pixels_again(self):
        panel = self.panel()
        self.App.processEvents()
        layer = self.viewer.open(self.path, plugin="napari")[0]
        data = layer.data
        with patch.object(tifffile.TiffPageSeries, "asarray", side_effect=AssertionError("Pixels reloaded")):
            self.App.processEvents()
        self.assertIs(layer.data, data)
        self.assert_calibrated(layer, panel)

    def test_existing_native_layer_keeps_pixel_edits_on_panel_startup(self):
        layer = self.viewer.open(self.path, plugin="napari")[0]
        edited = layer.data
        edited[...] = 17
        with patch.object(tifffile.TiffPageSeries, "asarray", side_effect=AssertionError("Pixels reloaded")):
            panel = self.panel()
            self.App.processEvents()
        self.assertIs(layer.data, edited)
        np.testing.assert_array_equal(layer.data, 17)
        self.assert_calibrated(layer, panel)

    def test_explicit_scale_and_metadata_are_preserved(self):
        layer = self.viewer.open(self.path, plugin="napari", scale=(1, .7, .3, .4))[0]
        panel = self.panel()
        self.App.processEvents()
        np.testing.assert_array_equal(layer.scale, (1, .7, .3, .4))
        self.assertFalse(layer.metadata)
        layer = self.viewer.open(self.path, plugin="napari",
                                 metadata={"unit": "nm", "dims": "TZYX", "custom": "keep"})[0]
        panel._maybe_normalize_dragdrop_layer(layer)
        self.assertEqual(layer.metadata, {"unit": "nm", "dims": "TZYX", "custom": "keep"})
        np.testing.assert_array_equal(layer.scale, (1, 1, 1, 1))

    def test_custom_transform_is_not_overwritten(self):
        layer = self.viewer.open(self.path, plugin="napari", translate=(0, 0, 1, 0))[0]
        self.panel()
        self.App.processEvents()
        np.testing.assert_array_equal(layer.scale, (1, 1, 1, 1))
        np.testing.assert_array_equal(layer.translate, (0, 0, 1, 0))
        self.assertFalse(layer.metadata)

    def test_programmatic_layer_with_source_path_is_not_reinterpreted(self):
        layer = self.viewer.add_image(np.ones(self.shape), rgb=False, metadata={"source": self.path})
        self.panel()
        self.App.processEvents()
        self.assertEqual(layer.metadata, {"source": self.path})
        np.testing.assert_array_equal(layer.scale, (1, 1, 1, 1))

    def test_other_readers_and_derived_layers_are_not_reinterpreted(self):
        from napari.layers._source import layer_source
        parent = self.viewer.add_image(np.zeros(self.shape), rgb=False)
        for provenance in ({"reader_plugin": "chosen-reader"},
                           {"reader_plugin": "napari", "parent": parent}):
            with layer_source(path=self.path, **provenance):
                layer = self.viewer.add_image(np.ones(self.shape), rgb=False)
            panel = self.panel()
            self.App.processEvents()
            self.assertFalse(layer.metadata)
            np.testing.assert_array_equal(layer.scale, (1, 1, 1, 1))
            panel.dispose()

    def test_removed_layer_is_not_annotated_by_pending_drop_callback(self):
        self.panel()
        self.App.processEvents()
        layer = self.viewer.open(self.path, plugin="napari")[0]
        self.viewer.layers.remove(layer)
        self.App.processEvents()
        self.assertFalse(layer.metadata)
        np.testing.assert_array_equal(layer.scale, (1, 1, 1, 1))

    def test_cropped_layer_cannot_borrow_original_geometry(self):
        layer = self.viewer.open(self.path, plugin="napari")[0]
        layer.data = layer.data[..., :-1]
        self.panel()
        self.App.processEvents()
        self.assertFalse(layer.metadata)
        np.testing.assert_array_equal(layer.scale, (1, 1, 1, 1))

    def test_unitless_default_tiff_does_not_invent_micrometer_calibration(self):
        tifffile.imwrite(self.path, np.zeros((8, 9), dtype=np.uint16), metadata={"axes": "YX"},
                         resolutionunit="NONE")
        self.assertIsNone(read_tiff_layer_calibration(self.path, (8, 9)))

    def test_ome_calibration_converts_units_without_reading_pixels(self):
        path = str(Path(self.path).with_name("ome.tif"))
        tifffile.imwrite(path, np.zeros(self.shape, np.uint16), ome=True, metadata={
            "axes": "TZYX", "PhysicalSizeX": 110, "PhysicalSizeXUnit": "nm",
            "PhysicalSizeY": 120, "PhysicalSizeYUnit": "nm",
            "PhysicalSizeZ": 200, "PhysicalSizeZUnit": "nm",
            "TimeIncrement": 2, "TimeIncrementUnit": "min"})
        with patch.object(tifffile.TiffPageSeries, "asarray", side_effect=AssertionError("Pixels reloaded")):
            calibration = read_tiff_layer_calibration(path, self.shape)
        np.testing.assert_allclose(calibration["scale"], (1, .2, .12, .11))
        self.assertEqual(calibration["metadata"]["time_interval"], 120)

    def test_multiple_series_and_channels_are_not_ambiguously_calibrated(self):
        path = str(Path(self.path).with_name("multiple.tif"))
        with tifffile.TiffWriter(path) as writer:
            for spacing in (.2, .4):
                writer.write(np.zeros((5, 8, 9), np.uint16), photometric="minisblack",
                             metadata={"axes": "ZYX", "spacing": spacing, "unit": "um"})
        self.assertIsNone(read_tiff_layer_calibration(path, (5, 8, 9)))
        path = str(Path(self.path).with_name("channels.tif"))
        tifffile.imwrite(path, np.zeros((2, 5, 8, 9), np.uint16), photometric="minisblack",
                         metadata={"axes": "CZYX", "spacing": .2, "unit": "um"})
        self.assertIsNone(read_tiff_layer_calibration(path, (2, 5, 8, 9)))

    def test_sigma_reader_calibration_is_not_changed(self):
        layer = self.viewer.open(self.path, plugin="napari-sigma")[0]
        metadata = dict(layer.metadata)
        data = layer.data
        panel = self.panel()
        self.App.processEvents()
        self.assertEqual(layer.metadata, metadata)
        self.assertIs(layer.data, data)
        self.assert_calibrated(layer, panel)

    def test_drop_then_open_twice_selects_new_pixels_and_calibration(self):
        from qtpy.QtWidgets import QFileDialog, QMessageBox
        panel = self.panel()
        self.App.processEvents()
        original = self.viewer.open(self.path, plugin="napari")[0]
        original_data = original.data
        self.App.processEvents()
        for number, spacing in enumerate((.4, .8), 1):
            path = str(Path(self.path).with_name(f"second-{number}.tif"))
            pixels = np.full(self.shape, number * 73, np.uint16)
            tifffile.imwrite(path, pixels, imagej=True, resolution=(5, 5), metadata={
                "axes": "TZYX", "unit": "um", "spacing": spacing})
            with patch.object(QFileDialog, "getOpenFileName", return_value=(path, "")) as dialog, \
                 patch.object(QMessageBox, "critical") as error:
                panel.open_btn.click()
                self.App.processEvents()
                dialog.assert_called_once()
                error.assert_not_called()
            newest = self.viewer.layers[-1]
            self.assertEqual(len(self.viewer.layers), number + 1)
            self.assertIs(panel._selected_info_raw_layer(), newest)
            self.assertIs(panel._selected_frangi_raw_layer(), newest)
            self.assertIs(panel._selected_segmentation_raw_layer(), newest)
            np.testing.assert_array_equal(newest.data, pixels)
            np.testing.assert_allclose(newest.scale, (1, spacing, .2, .2))
            self.assertIn(f"vz={spacing}", panel.lbl_voxpix.text())
            self.assertTrue(panel.open_btn.isEnabled())
        self.assertIs(original.data, original_data)

    def test_open_recovers_after_cancel_and_load_error(self):
        from qtpy.QtWidgets import QFileDialog, QMessageBox
        panel = self.panel()
        self.App.processEvents()
        for path in ("", str(Path(self.path).with_name("missing.tif")), self.path, self.path):
            with patch.object(QFileDialog, "getOpenFileName", return_value=(path, "")), \
                 patch.object(QMessageBox, "critical") as error:
                panel.open_btn.click()
                self.App.processEvents()
                self.assertEqual(error.call_count, int(path.endswith("missing.tif")))
            self.assertTrue(panel.open_btn.isEnabled())
        self.assertEqual(len(self.viewer.layers), 2)
        self.assertIs(panel._selected_info_raw_layer(), self.viewer.layers[-1])
        self.assert_calibrated(self.viewer.layers[-1], panel)


if __name__ == "__main__":
    unittest.main()
