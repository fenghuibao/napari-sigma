from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
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


build, install, launch = (module(name) for name in ("build", "install", "launch"))


class InstallerTests(unittest.TestCase):
    def test_platform_versions(self):
        intel = build.requirements("osx-64")
        self.assertIn("numpy==1.26.4", intel)
        self.assertIn("torch==2.2.2", intel)
        self.assertIn("torch==2.13.0+cpu", build.requirements("win-64"))
        self.assertIn("torch==2.13.0", build.requirements("osx-arm64"))

    def test_shortcuts_are_isolated_and_nonterminal(self):
        menu = build.menu_metadata("0.0.5")
        json.dumps(menu)
        item = menu["menu_items"][0]
        self.assertFalse(item["activate"])
        self.assertFalse(item["terminal"])
        self.assertEqual(item["command"][:2], ["{{ PYTHON }}", "-I"])
        self.assertEqual(item["platforms"]["win"]["command"][:2], ["{{ PYTHONW }}", "-I"])
        self.assertEqual(item["platforms"]["osx"]["LSMinimumSystemVersion"], "14.0")

    def test_constructor_does_not_modify_shell_or_register_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payload.txt").write_text("payload")
            for target in build.PLATFORMS:
                config = build.constructor_config(target, root / "runtime", root)
                self.assertFalse(config["initialize_conda"])
                self.assertFalse(config["register_envs"])
                self.assertEqual(config["menu_packages"], [])
                self.assertIn("unsigned", config["installer_filename"])
                self.assertEqual(list(config["extra_files"][0].values()), ["sigma-desktop/payload.txt"] if os.name != "nt" else ["sigma-desktop\\payload.txt"])
                if target == "win-64":
                    self.assertFalse(config["register_python"])
                else:
                    self.assertEqual(config["pkg_name"], "sigma-0.0.5")

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

    def test_installer_cannot_modify_unrelated_python(self):
        with self.assertRaisesRegex(RuntimeError, "own bundled Python"):
            install.install_prefix()

    def test_caches_are_per_user(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(launch, "user_directories", return_value=(root / "logs", root / "cache")), patch.dict(os.environ, {}, clear=True):
                log = launch.prepare_process()
                self.assertEqual(log, root / "logs/desktop.log")
                self.assertTrue(Path(os.environ["MPLCONFIGDIR"]).is_dir())
                self.assertTrue(Path(os.environ["NUMBA_CACHE_DIR"]).is_dir())


if __name__ == "__main__":
    unittest.main()
