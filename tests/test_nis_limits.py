from io import BytesIO
import struct
import unittest
from unittest.mock import patch
import zlib

from napari_sigma import _nis_tiff as nis


class NisMetadataLimits(unittest.TestCase):
    @staticmethod
    def compressed(payload):
        return BytesIO(bytes((nis._CLX_COMPRESSED, 0)) + bytes(10) + zlib.compress(payload))

    def test_valid_compressed_scalar(self):
        payload = bytes((nis._CLX_INT32, 1)) + "x".encode("utf-16le") + struct.pack("<i", 7)
        self.assertEqual(nis._decode_clx_items(self.compressed(payload)), {"x": 7})

    def test_expansion_is_bounded_before_parsing(self):
        with patch.object(nis, "_MAX_METADATA_BYTES", 1024):
            with self.assertRaisesRegex(nis._InvalidNisMetadata, "limit"):
                nis._decode_clx_items(self.compressed(b"a" * 2048))

    def test_total_nested_expansion_budget(self):
        payload = bytes((nis._CLX_BYTEARRAY, 0)) + struct.pack("<Q", 800) + bytes(800)
        nested = self.compressed(payload).getvalue()
        with patch.object(nis, "_MAX_METADATA_BYTES", len(payload) + 4):
            with self.assertRaisesRegex(nis._InvalidNisMetadata, "limit"):
                nis._decode_clx_items(self.compressed(nested))

    def test_item_and_length_limits(self):
        with self.assertRaises(nis._InvalidNisMetadata):
            nis._decode_clx_items(BytesIO(), count=nis._MAX_METADATA_ITEMS + 1)
        with self.assertRaises(nis._InvalidNisMetadata):
            nis._read_exact(BytesIO(), nis._MAX_METADATA_BYTES + 1)
