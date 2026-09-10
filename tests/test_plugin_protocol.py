"""Exercise installed manifest commands through npe2, not only direct helpers."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from npe2 import PluginManager, io_utils
from npe2.manifest import PluginManifest
from PIL import Image
import tifffile

import napari_sigma
from napari_sigma._reader import napari_get_reader


class PluginProtocolRegressions(unittest.TestCase):
    def setUp(self):
        manifest_path = Path(napari_sigma.__file__).with_name("napari.yaml")
        self.manager = PluginManager()
        self.manager.register(PluginManifest.from_file(manifest_path))
        self.manager_patch = patch.object(PluginManager, "instance", return_value=self.manager)
        self.manager_patch.start()
        self.addCleanup(self.manager_patch.stop)
        self.addCleanup(self.manager.unregister, "napari-sigma")

    def test_factory_accepts_protocol_keyword(self):
        for path in ("example.tif", Path("example.tif"), ["example.tif"]):
            with self.subTest(path=path):
                self.assertTrue(callable(napari_get_reader(path=path)))
        for path in ("example.txt", [], ["first.tif", "second.tif"]):
            with self.subTest(path=path):
                self.assertIsNone(napari_get_reader(path=path))

    def test_reader_dispatch_for_supported_formats(self):
        expected = np.arange(72, dtype=np.uint8).reshape(8, 9)
        with tempfile.TemporaryDirectory() as tmp:
            for suffix in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                path = Path(tmp) / f"图像{suffix}"
                if suffix in {".tif", ".tiff"}:
                    tifffile.imwrite(path, expected, photometric="minisblack")
                    reference = expected
                else:
                    Image.fromarray(expected).save(path)
                    with Image.open(path) as image:
                        reference = np.asarray(image)
                for stack in (False, True):
                    with self.subTest(suffix=suffix, stack=stack):
                        layers, reader = io_utils.read_get_reader(
                            [path], stack=stack, plugin_name="napari-sigma")
                        self.assertEqual(reader.command, "napari-sigma.reader")
                        self.assertEqual(layers[0][2], "image")
                        np.testing.assert_array_equal(layers[0][0], reference)

    def test_writer_and_reader_dispatch_preserve_labels(self):
        expected = np.full((2, 3, 8, 9), 2**40 + 1, dtype=np.uint64)
        attributes = {"name": "labels", "scale": (1, 2, .3, .2),
                      "metadata": {"dims": "TZYX", "unit": "um"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.tif"
            written, writer = io_utils.write_get_writer(
                path, [(expected, attributes, "labels")], plugin_name="napari-sigma")
            self.assertEqual(writer.command, "napari-sigma.write_labels")
            self.assertEqual(written, [str(path)])
            layers, _ = io_utils.read_get_reader(path, plugin_name="napari-sigma")
            actual, kwargs, kind = layers[0]
            self.assertEqual(kind, "labels")
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_allclose(kwargs["scale"], attributes["scale"])

    def test_writer_and_reader_dispatch_preserve_native_rgb(self):
        expected = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
        attributes = {"name": "rgb", "rgb": True, "scale": (.3, .2), "metadata": {}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rgb.tif"
            written, writer = io_utils.write_get_writer(
                path, [(expected, attributes, "image")], plugin_name="napari-sigma")
            self.assertEqual(writer.command, "napari-sigma.write_single")
            self.assertEqual(written, [str(path)])
            layers, _ = io_utils.read_get_reader(path, plugin_name="napari-sigma")
            self.assertEqual(len(layers), 3)
            for channel, (actual, kwargs, kind) in enumerate(layers):
                self.assertEqual(kind, "image")
                np.testing.assert_array_equal(actual, expected[..., channel])
                np.testing.assert_allclose(kwargs["scale"], attributes["scale"])
