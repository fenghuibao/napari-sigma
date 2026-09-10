"""Pixel-exact TIFF axes and RGB round trips, including native napari metadata."""
from itertools import permutations
from pathlib import Path
import unittest

import numpy as np
import tifffile
from _fixtures import temporary_directory

from napari_sigma._image_io import _normalize_tiff_data_to_tczyx, load_image_tc_zyx
from napari_sigma._reader import napari_get_reader
from napari_sigma._writer import write_single_image


class TiffAxesRegressions(unittest.TestCase):
    def test_rgb_stacks_keep_channels_separate_from_z_and_time(self):
        with temporary_directory(self) as tmp:
            for axes, shape in (("ZYXS", (5, 8, 9, 3)),
                                ("TYXS", (5, 8, 9, 3)),
                                ("TZYXS", (2, 5, 8, 9, 3))):
                with self.subTest(axes=axes):
                    raw = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
                    path = str(Path(tmp) / f"{axes}.tif")
                    tifffile.imwrite(path, raw, photometric="rgb", metadata={"axes": axes})
                    actual, meta = load_image_tc_zyx(path)
                    if axes == "ZYXS":
                        expected = np.moveaxis(raw, -1, 0)[None]
                    elif axes == "TYXS":
                        expected = np.moveaxis(raw, -1, 1)[:, :, None]
                    else:
                        expected = np.moveaxis(raw, -1, 1)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(meta["channel_names"], ["Red", "Green", "Blue"])
                    layers = napari_get_reader(path=path)(path)
                    self.assertEqual(len(layers), 3)
                    for channel, (array, _, _) in enumerate(layers):
                        np.testing.assert_array_equal(array, raw[..., channel])

    def test_native_rgb_writer_uses_spatial_scale_without_sample_axis(self):
        with temporary_directory(self) as tmp:
            for shape, scale, axes in (((8, 9, 3), (.3, .2), "YXS"),
                                      ((5, 8, 9, 3), (2, .3, .2), "ZYXS"),
                                      ((2, 5, 8, 9, 3), (1, 2, .3, .2), "TZYXS")):
                with self.subTest(axes=axes):
                    raw = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
                    path = str(Path(tmp) / f"native-{axes}.tif")
                    write_single_image(path, raw, {"rgb": True, "scale": scale, "metadata": {}})
                    with tifffile.TiffFile(path) as tif:
                        self.assertEqual(tif.series[0].axes, axes)
                        self.assertEqual(tif.pages[0].photometric, tifffile.PHOTOMETRIC.RGB)
                    layers = napari_get_reader(path=path)(path)
                    self.assertEqual(len(layers), 3)
                    for channel, (array, kwargs, _) in enumerate(layers):
                        np.testing.assert_array_equal(array, raw[..., channel])
                        np.testing.assert_allclose(kwargs["scale"], scale)

    def test_rgb_movie_writer_roundtrip_accepts_legacy_and_spatial_scales(self):
        raw = np.arange(5 * 8 * 9 * 3, dtype=np.uint16).reshape(5, 8, 9, 3)
        with temporary_directory(self) as tmp:
            for dims in ("TYXC", "TYXS", "TYX"):
                for scale in ((1, .3, .2), (1, .3, .2, 1)):
                    with self.subTest(dims=dims, scale=scale):
                        # Do not overwrite another case's still-mapped file.
                        path = str(Path(tmp) / f"movie-{dims}-{len(scale)}.tif")
                        write_single_image(path, raw, {
                            "rgb": True, "scale": scale,
                            "metadata": {"dims": dims, "time_interval": 7}})
                        layers = napari_get_reader(path=path)(path)
                        self.assertEqual(len(layers), 3)
                        for channel, (array, kwargs, _) in enumerate(layers):
                            np.testing.assert_array_equal(array, raw[..., channel])
                            np.testing.assert_allclose(kwargs["scale"], (1, .3, .2))
                            self.assertEqual(kwargs["metadata"]["time_interval"], 7)

    def test_all_explicit_tcz_permutations_preserve_pixels_and_views(self):
        expected = np.arange(2 * 4 * 5 * 8 * 9, dtype=np.uint16).reshape(2, 4, 5, 8, 9)
        for prefix in permutations("TCZ"):
            axes = "".join(prefix) + "YX"
            raw = expected.transpose(tuple("TCZYX".index(axis) for axis in axes))
            raw.flags.writeable = False
            with self.subTest(axes=axes):
                actual, names, dims = _normalize_tiff_data_to_tczyx(raw, axes)
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(np.shares_memory(actual, raw))
                self.assertEqual(len(names), 4)
                self.assertEqual(dims, "TCZYX")

    def test_ome_ctzyx_roundtrip_including_physical_metadata(self):
        raw = np.arange(4 * 2 * 5 * 8 * 9, dtype=np.uint16).reshape(4, 2, 5, 8, 9)
        with temporary_directory(self) as tmp:
            path = str(Path(tmp) / "channels.ome.tif")
            tifffile.imwrite(path, raw, ome=True, photometric="minisblack", metadata={
                "axes": "CTZYX", "PhysicalSizeX": .2, "PhysicalSizeY": .3,
                "PhysicalSizeZ": 2., "TimeIncrement": 7.})
            actual, meta = load_image_tc_zyx(path)
            np.testing.assert_array_equal(actual, np.moveaxis(raw, 0, 1))
            np.testing.assert_allclose(meta["scale_per_axis"], (1, 1, 2, .3, .2))
            self.assertEqual(meta["time_interval"], 7.)

    def test_unannotated_page_axis_is_explicitly_assumed_z(self):
        raw = np.arange(5 * 8 * 9, dtype=np.uint16).reshape(5, 8, 9)
        with temporary_directory(self) as tmp:
            path = str(Path(tmp) / "pages.tif")
            tifffile.imwrite(path, raw, photometric="minisblack", metadata=None)
            actual, meta = load_image_tc_zyx(path)
            self.assertIn(meta["axes"], {"IYX", "QYX"})
            self.assertEqual(meta["axis_assumptions"], {meta["axes"][0]: "Z"})
            np.testing.assert_array_equal(actual, raw[None, None])
        for axes in ("IYX", "QYX"):
            actual, _, dims = _normalize_tiff_data_to_tczyx(raw, axes)
            np.testing.assert_array_equal(actual, raw[None, None])
            self.assertEqual(dims, "ZYX")
            actual, _, dims = _normalize_tiff_data_to_tczyx(raw, axes, {"frames": 5})
            np.testing.assert_array_equal(actual, raw[:, None, None])
            self.assertEqual(dims, "TYX")

    def test_gray_rgb_and_alpha_conventions_are_preserved(self):
        gray = np.arange(5 * 8 * 9, dtype=np.uint16).reshape(5, 8, 9)
        for axes in ("ZYXS", "TYXS"):
            for samples in (3, 4):
                rgb = np.repeat(gray[..., None], samples, axis=-1)
                if samples == 4:
                    rgb[..., -1] = 65535
                with self.subTest(axes=axes, samples=samples):
                    actual, names, dims = _normalize_tiff_data_to_tczyx(rgb, axes)
                    expected = gray[None, None] if axes[0] == "Z" else gray[:, None, None]
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(names, ["Channel 1"])
                    self.assertEqual(dims, axes[:-1])
        rgba = np.stack([gray, gray + 1, gray + 2, gray * 0 + 65535], axis=-1)
        actual, _, _ = _normalize_tiff_data_to_tczyx(rgba, "ZYXS")
        np.testing.assert_array_equal(actual, np.moveaxis(rgba[..., :3], -1, 0)[None])

    def test_invalid_or_ambiguous_axes_rejected(self):
        for axes, shape in (("QQYX", (2, 5, 8, 9)), ("TYXX", (2, 5, 8, 9)),
                            ("ABYX", (2, 5, 8, 9)), ("YX", (2, 8, 9))):
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                _normalize_tiff_data_to_tczyx(np.zeros(shape, np.uint8), axes)

    def test_samples_and_logical_channels_do_not_drop_or_mix_signal(self):
        raw = np.arange(2 * 4 * 5 * 8 * 9 * 3, dtype=np.uint16).reshape(2, 4, 5, 8, 9, 3)
        expected = raw.transpose(0, 1, 5, 2, 3, 4).reshape(2, 12, 5, 8, 9)
        for axes in ("TCZYXS", "STCZYX", "CTZSYX"):
            array = raw.transpose(tuple("TCZYXS".index(axis) for axis in axes))
            with self.subTest(axes=axes):
                actual, names, dims = _normalize_tiff_data_to_tczyx(array, axes)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(len(names), 12)
                self.assertEqual(dims, "TCZYX")
        # A microscopy C axis must retain all channels, even when more than RGB.
        raw = np.arange(8 * 9 * 5, dtype=np.uint16).reshape(8, 9, 5)
        actual, names, _ = _normalize_tiff_data_to_tczyx(raw, "YXC")
        np.testing.assert_array_equal(actual, np.moveaxis(raw, -1, 0)[None, :, None])
        self.assertEqual(len(names), 5)

    def test_legacy_rgb_movie_without_native_flag_still_roundtrips(self):
        raw = np.arange(2 * 8 * 9 * 3, dtype=np.uint16).reshape(2, 8, 9, 3)
        with temporary_directory(self) as tmp:
            path = str(Path(tmp) / "legacy.tif")
            write_single_image(path, raw, {"metadata": {"dims": "TYXC"}, "scale": (1, .3, .2, 1)})
            actual, _ = load_image_tc_zyx(path)
            np.testing.assert_array_equal(actual, np.moveaxis(raw, -1, 1)[:, :, None])
