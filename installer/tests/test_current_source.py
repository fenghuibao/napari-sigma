from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_current_macos


class CurrentSourceTests(unittest.TestCase):
    def fixture(self, directory, content=b'title = "Current wording"\n', extra=False):
        root = Path(directory)
        source = root / "src/napari_sigma"
        source.mkdir(parents=True)
        (source / "_widget.py").write_bytes(b'title = "Current wording"\n')
        wheel = root / "napari_sigma-0.0.6-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("napari_sigma-0.0.6.dist-info/METADATA", "Name: napari-sigma\nVersion: 0.0.6\n")
            archive.writestr("napari_sigma/_widget.py", content)
            if extra:
                archive.writestr("napari_sigma/stale.py", "")
        return root, wheel

    def test_current_wording_is_preserved_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            source, wheel = self.fixture(directory)
            version, files = build_current_macos.verify_source_wheel(source, wheel)
            self.assertEqual(version, "0.0.6")
            self.assertEqual(set(files), {"napari_sigma/_widget.py"})

    def test_old_wording_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, wheel = self.fixture(directory, b'title = "Old wording"\n')
            with self.assertRaisesRegex(ValueError, "differs from current source"):
                build_current_macos.verify_source_wheel(source, wheel)

    def test_stale_source_in_wheel_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, wheel = self.fixture(directory, extra=True)
            with self.assertRaisesRegex(ValueError, "stale source"):
                build_current_macos.verify_source_wheel(source, wheel)
