from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import tifffile
from _fixtures import temporary_directory

from napari_sigma._image_io import load_image_tc_zyx
from napari_sigma._metadata import layer_dims_tag
from napari_sigma._reader import napari_get_reader, tczyx_to_layer_data
from napari_sigma._writer import write_single_image, write_single_labels


class IORegressions(unittest.TestCase):
    def test_mp4_export_with_headless_opencv(self):
        import cv2
        frames = np.zeros((3, 16, 20, 3), dtype=np.uint8)
        frames[..., 0] = 180
        with temporary_directory(self) as tmp:
            path = str(Path(tmp) / "movie.mp4")
            write_single_image(path, frames, {"metadata": {"fps": 5}})
            capture = cv2.VideoCapture(path)
            try:
                self.assertTrue(capture.isOpened())
                count = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    self.assertEqual(frame.shape, (16, 20, 3))
                    self.assertGreater(float(frame[..., 2].mean()), 160)
                    self.assertLess(float(frame[..., :2].mean()), 15)
                    count += 1
                self.assertEqual(count, len(frames))
            finally:
                capture.release()

    def test_label_writer_protocol_and_large_ids(self):
        with temporary_directory(self) as tmp:
            path = str(Path(tmp) / "labels.tif")
            data = np.full((8, 9), 2**40 + 1, np.uint64)
            write_single_labels(path, data, {"scale": [.3, .2]})
            actual, _, kind = napari_get_reader(path)(path)[0]
            self.assertEqual(kind, "labels")
            np.testing.assert_array_equal(actual, data)

    def test_roundtrip_axes_labels_and_calibration(self):
        with temporary_directory(self) as tmp:
            for dims, shape, scale in (("YX", (8, 9), (.3, .2)),
                    ("ZYX", (3, 8, 9), (2, .3, .2)),
                    ("TYX", (2, 8, 9), (1, .3, .2)),
                    ("TZYX", (2, 3, 8, 9), (1, 2, .3, .2))):
                for value in (1, 70000):
                    with self.subTest(dims=dims, value=value):
                        raw = np.full(shape, value, np.uint32)
                        path = str(Path(tmp) / f"{dims}-{value}.tif")
                        meta = dict(layer_type="labels", scale=np.asarray(scale), metadata={
                            "dims": dims, "unit": "um", "time_interval": 7., "is_frangi": False})
                        write_single_image(path, raw, meta)
                        data, kwargs, kind = napari_get_reader(path)(path)[0]
                        self.assertEqual(kind, "labels")
                        self.assertEqual(kwargs["metadata"]["dims"], dims)
                        np.testing.assert_array_equal(data, raw)
                        np.testing.assert_allclose(kwargs["scale"], scale)
                        if "T" in dims:
                            self.assertEqual(kwargs["metadata"]["time_interval"], 7.)

    def test_multichannel_imagej_reorders_data_and_spacing(self):
        raw = np.arange(2 * 2 * 3 * 8 * 9, dtype=np.uint16).reshape(2, 2, 3, 8, 9)
        with temporary_directory(self) as tmp:
            path = str(Path(tmp) / "channels.tif")
            write_single_image(path, raw, dict(scale=(1, 1, 2, .3, .2), metadata={"dims": "TCZYX"}))
            data, meta = load_image_tc_zyx(path)
            np.testing.assert_array_equal(data, raw)
            np.testing.assert_allclose(meta["scale_per_axis"], (1, 1, 2, .3, .2))
            layers = tczyx_to_layer_data(data, meta, "channels")
            self.assertEqual(len(layers), 2)
            for index, (data, kwargs, kind) in enumerate(layers):
                np.testing.assert_array_equal(data, raw[:, index])
                self.assertEqual(kwargs["metadata"]["dims"], "TZYX")

    def test_non_imagej_custom_metadata_and_units(self):
        with temporary_directory(self) as tmp:
            path = str(Path(tmp) / "calibration.tif")
            raw = np.full((3, 8, 9), 70000, np.uint32)
            write_single_image(path, raw, dict(scale=(.002, .0003, .0002), layer_type="labels",
                metadata={"dims": "ZYX", "unit": "mm", "sigma_layer_role": "structural_response"}))
            _, meta = load_image_tc_zyx(path)
            self.assertTrue(meta["is_frangi"])
            np.testing.assert_allclose(meta["scale_per_axis"][-3:], (2, .3, .2))

    def test_reader_does_not_import_ui_plotting_or_torch(self):
        code = "import sys; from napari_sigma._reader import napari_get_reader; assert not any(x in sys.modules for x in ('qtpy', 'matplotlib', 'torch', 'napari_sigma._widget'))"
        subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy(), timeout=30)

    def test_shared_dims_precedence(self):
        self.assertEqual(layer_dims_tag(SimpleNamespace(metadata={"dims": "ZYX", "dims_out": "TZYX"})), "TZYX")

    def test_legacy_shaped_calibration_takes_precedence_over_inch_default(self):
        with temporary_directory(self) as tmp:
            path = str(Path(tmp) / "legacy.tif")
            tifffile.imwrite(path, np.full((3, 8, 9), 70000, np.uint32),
                resolution=(5, 10 / 3), metadata={"axes": "ZYX", "unit": "um", "spacing": 2, "layer_type": "labels"})
            _, meta = load_image_tc_zyx(path)
            np.testing.assert_allclose(meta["scale_per_axis"][-3:], (2, .3, .2))
