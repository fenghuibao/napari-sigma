from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_windows_exe as win
import verify_windows_icons as icons
import build


class SingleExeTests(unittest.TestCase):
    def test_source_and_cache_versions_are_explicit(self):
        self.assertEqual(build.VERSION, '0.0.6')
        self.assertEqual(build.WINDOWS_TORCH, '2.13.0+cu130')
        self.assertEqual(win.PYTHON['version'], '3.13.15')
        for record in (win.PYTHON, win.INNO):
            self.assertTrue(record['url'].startswith('https://github.com/'))
            self.assertRegex(record['sha256'], r'^[0-9a-f]{64}$')

    def test_dependency_cache_requires_exact_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'cache.zip'
            package.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                win.extract_wheels(package, Path(directory) / 'wheels')
            self.assertFalse((Path(directory) / 'wheels').exists())

    def test_extracts_only_flat_wheels_not_old_setup_or_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'cache.zip'
            with zipfile.ZipFile(package, 'w') as archive:
                archive.writestr('SIGMA-Setup.exe', b'old-incomplete-setup')
                archive.writestr('wheelhouse/torch.whl', b'cuda')
                archive.writestr('wheelhouse/../../escape.whl', b'no')
                archive.writestr('unrelated.whl', b'no')
            with patch.object(win, 'CACHE_SHA256', win.digest(package)):
                win.extract_wheels(package, root / 'wheels')
            self.assertEqual([p.name for p in (root / 'wheels').iterdir()], ['torch.whl'])
            self.assertEqual((root / 'wheels/torch.whl').read_bytes(), b'cuda')
            self.assertFalse((root / 'SIGMA-Setup.exe').exists())

    def test_installer_is_self_contained_and_non_destructive(self):
        script = (ROOT / 'windows_installer.iss').read_text()
        for directive in ('DiskSpanning=no', 'SolidCompression=yes', 'Compression=lzma2/ultra64',
                          'PrivilegesRequired=lowest', 'AppVerName=SIGMA', 'UninstallDisplayName=SIGMA',
                          'SetupIconFile=', 'UninstallDisplayIcon=', 'IconFilename:', '-I -B',
                          'function PrepareToInstall', 'The selected folder is not empty'):
            self.assertIn(directive, script)
        self.assertNotIn('filesandordirs', script)
        self.assertNotIn('external', script.lower())
        self.assertNotIn('download', script.split('[Setup]', 1)[1].lower())
        self.assertNotIn('[Registry]', script)

    def test_multiresolution_icon_uses_existing_artwork(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            build.make_icons(path)
            self.assertEqual((path / 'sigma.png').read_bytes(), (ROOT / 'assets/sigma-logo.png').read_bytes())
            self.assertEqual(len(icons.ico_images(path / 'sigma.ico')), 5)

    def test_pip_environment_does_not_inherit_user_configuration(self):
        with patch.dict('os.environ', {'PIP_INDEX_URL':'https://unexpected.invalid', 'PYTHONPATH':'poison'}):
            env = win.pip_environment(Path('work'))
        self.assertNotIn('PIP_INDEX_URL', env)
        self.assertNotIn('PYTHONPATH', env)
        self.assertEqual(env['MPL_IGNORE_SYSTEM_FONTS'], '1')


if __name__ == '__main__':
    unittest.main()
