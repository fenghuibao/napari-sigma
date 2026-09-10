from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(f"desktop_{name}", ROOT / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


build, install, launch, verify = (module(name) for name in ("build", "install", "launch", "verify"))


class InstallerTests(unittest.TestCase):
    def test_platform_versions(self):
        intel = build.requirements("osx-64")
        self.assertIn("numpy==1.26.4", intel)
        self.assertIn("torch==2.2.2", intel)
        self.assertIn("tifffile==2026.3.3", intel)
        self.assertIn("torch==2.13.0+cpu", build.requirements("win-64"))
        self.assertIn("torch==2.13.0", build.requirements("osx-arm64"))

    def test_smoke_fixture_lives_until_child_exit_and_is_cleaned(self):
        observed = []
        def child(command, **kwargs):
            directory = Path(command[command.index("--smoke-data-dir") + 1])
            self.assertTrue(directory.is_dir())
            (directory / "labels.tif").write_bytes(b"fixture")
            observed.append(directory)
            return subprocess.CompletedProcess(command, 0, b"ok")
        with patch.object(verify.subprocess, "run", side_effect=child):
            result = verify.run_smoke(Path(sys.executable), ROOT, ROOT, {})
        self.assertEqual(result.returncode, 0)
        self.assertFalse(observed[0].exists())

    def test_smoke_fixture_is_cleaned_on_child_timeout(self):
        observed = []
        def child(command, **kwargs):
            observed.append(Path(command[command.index("--smoke-data-dir") + 1]))
            raise subprocess.TimeoutExpired(command, 1200)
        with patch.object(verify.subprocess, "run", side_effect=child):
            with self.assertRaises(subprocess.TimeoutExpired):
                verify.run_smoke(Path(sys.executable), ROOT, ROOT, {})
        self.assertFalse(observed[0].exists())

    def test_regression_fixture_outlives_local_memory_map(self):
        import mmap
        spec = importlib.util.spec_from_file_location("fixtures", ROOT.parent / "tests/_fixtures.py")
        fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixtures)
        observed = []
        class FixtureCase(unittest.TestCase):
            def runTest(case):
                with fixtures.temporary_directory(case) as directory:
                    path = Path(directory) / "mapped.bin"
                    path.write_bytes(b"data")
                    with path.open("r+b") as stream:
                        mapped = mmap.mmap(stream.fileno(), 0)
                case.assertTrue(path.exists())
                case.assertEqual(mapped[:], b"data")
                observed.append(Path(directory))
                # Let the local map go out of scope, as real test arrays do.
        result = unittest.TestResult()
        FixtureCase().run(result)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        self.assertFalse(observed[0].exists())

    def test_shortcuts_are_isolated_and_nonterminal(self):
        menu = build.menu_metadata("0.0.5")
        json.dumps(menu)
        item = menu["menu_items"][0]
        self.assertEqual(item["name"], "SIGMA")
        self.assertFalse(item["activate"])
        self.assertFalse(item["terminal"])
        self.assertEqual(item["command"][:2], ["{{ PYTHON }}", "-I"])
        self.assertEqual(item["platforms"]["win"]["command"][:2], ["{{ PYTHONW }}", "-I"])
        self.assertEqual(item["platforms"]["osx"]["LSMinimumSystemVersion"], "14.0")

    def test_macos_plist_does_not_duplicate_menuinst_properties(self):
        from menuinst.platforms.base import menuitem_defaults
        mac = build.menu_metadata("0.0.5")["menu_items"][0]["platforms"]["osx"]
        self.assertFalse(set(mac.get("info_plist_extra", {})) & set(menuitem_defaults["platforms"]["osx"]))
        self.assertEqual(mac["CFBundleDisplayName"], "SIGMA")

    @unittest.skipUnless(sys.platform == "darwin", "macOS bundle integration")
    def test_macos_bundle_plist_is_actually_generated(self):
        from menuinst.api import _load
        from menuinst.platforms.osx import MacOSMenuItem
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, items = _load(build.menu_metadata("0.0.5"), str(root), str(root), "user")
            item = items[0]
            with patch.object(MacOSMenuItem, "_base_location", return_value=root):
                item._create_application_tree()
                item._write_plistinfo()
                with (item.location / "Contents/Info.plist").open("rb") as stream:
                    plist = plistlib.load(stream)
                self.assertEqual(plist["CFBundleDisplayName"], "SIGMA")
                self.assertEqual(plist["CFBundleName"], "SIGMA")
                self.assertEqual(item.location.name, "SIGMA.app")
                self.assertEqual(plist["CFBundleIdentifier"], "org.fenghuibao.sigma.desktop")
                self.assertEqual(plist["CFBundleVersion"], "0.0.5")

    def test_constructor_does_not_modify_shell_or_register_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payload.txt").write_text("payload")
            for target in build.PLATFORMS:
                config = build.constructor_config(target, root / "runtime", root)
                self.assertFalse(config["initialize_conda"])
                self.assertFalse(config["register_envs"])
                self.assertEqual(config["menu_packages"], [])
                self.assertIn({"hash": {"algorithm": "sha256"}}, config["build_outputs"])
                self.assertIn("unsigned", config["installer_filename"])
                self.assertEqual(list(config["extra_files"][0].values()), ["sigma-desktop/payload.txt"] if os.name != "nt" else ["sigma-desktop\\payload.txt"])
                if target == "win-64":
                    self.assertFalse(config["register_python"])
                    self.assertEqual(config["uninstall_name"], "SIGMA")
                else:
                    self.assertEqual(config["pkg_name"], "sigma-0.0.5")

    def test_icons_use_supplied_artwork_in_all_native_formats(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            build.make_icons(destination)
            source = ROOT / "assets/sigma-logo.png"
            self.assertEqual((destination / "sigma.png").read_bytes(), source.read_bytes())
            with Image.open(source) as original:
                expected = original.convert("RGBA").resize((1024, 1024), Image.Resampling.LANCZOS)
            with Image.open(destination / "sigma.icns") as mac:
                mac.size = (1024, 1024)
                self.assertEqual(mac.convert("RGBA").tobytes(), expected.tobytes())
            with Image.open(destination / "sigma.ico") as win:
                self.assertEqual(win.ico.sizes(), {(16, 16), (32, 32), (48, 48), (64, 64), (256, 256)})
                self.assertEqual(win.convert("RGBA").tobytes(), expected.resize((256, 256), Image.Resampling.LANCZOS).tobytes())

    def test_hash_lock_includes_every_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("napari_sigma", "torch"):
                with zipfile.ZipFile(root / f"{name}-1-py3-none-any.whl", "w") as archive:
                    archive.writestr(f"{name}-1.dist-info/METADATA", f"Name: {name}\nVersion: 1\n")
                    archive.writestr("vendor/nested-1.dist-info/METADATA", "Name: nested\nVersion: 1\n")
            records = build.lock_wheels(root, root / "requirements.lock")
            self.assertEqual({r["name"] for r in records}, {"napari-sigma", "torch"})
            for record in records:
                digest = hashlib.sha256((root / record["filename"]).read_bytes()).hexdigest()
                self.assertEqual(record["sha256"], digest)
                self.assertIn(f"--hash=sha256:{digest}", (root / "requirements.lock").read_text())

    def test_incomplete_wheelhouse_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "missing"):
                build.lock_wheels(root, root / "requirements.lock")

    def test_cached_wheelhouse_requires_unchanged_hashes_and_platform(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheels = root / "wheelhouse"
            wheels.mkdir()
            for name in ("napari_sigma", "torch"):
                with zipfile.ZipFile(wheels / f"{name}-1-py3-none-any.whl", "w") as archive:
                    archive.writestr(f"{name}-1.dist-info/METADATA", f"Name: {name}\nVersion: 1\n")
            records = build.lock_wheels(wheels, root / "requirements.lock")
            bundle = {"platform": "osx-arm64", "sigma_version": build.VERSION, "wheels": records}
            (root / "bundle.json").write_text(json.dumps(bundle))
            self.assertEqual(build.validated_cached_wheels(root, "osx-arm64"), records)
            with self.assertRaisesRegex(ValueError, "different platform"):
                build.validated_cached_wheels(root, "win-64")
            with zipfile.ZipFile(wheels / "torch-1-py3-none-any.whl", "a") as archive:
                archive.writestr("unexpected.txt", "modified")
            with self.assertRaisesRegex(ValueError, "changed"):
                build.validated_cached_wheels(root, "osx-arm64")

    def test_installer_cannot_modify_unrelated_python(self):
        with self.assertRaisesRegex(RuntimeError, "own bundled Python"):
            install.install_prefix()

    def test_user_pip_configuration_cannot_redirect_install(self):
        with patch.dict(os.environ, {"PIP_TARGET": "/outside", "PIP_PREFIX": "/outside", "PIP_CONFIG_FILE": "/user/pip.ini"}):
            env = install.pip_environment()
            self.assertNotIn("PIP_TARGET", env)
            self.assertNotIn("PIP_PREFIX", env)
            self.assertEqual(env["PIP_CONFIG_FILE"], os.devnull)

    def test_caches_are_per_user(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(launch, "user_directories", return_value=(root / "logs", root / "cache")), patch.object(launch, "user_configuration", return_value=root / "settings"), patch.dict(os.environ, {}, clear=True):
                log = launch.prepare_process()
                self.assertEqual(log, root / "logs/desktop.log")
                self.assertTrue(Path(os.environ["MPLCONFIGDIR"]).is_dir())
                self.assertTrue(Path(os.environ["NUMBA_CACHE_DIR"]).is_dir())
                self.assertEqual(os.environ["NAPARI_CONFIG"], str(root / "settings/napari.yaml"))


if __name__ == "__main__":
    unittest.main()
