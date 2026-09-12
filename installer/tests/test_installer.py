from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import shutil
import struct
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
mac_icon = module("mac_icon")
sys.modules.setdefault("mac_icon", mac_icon)
mac_app = module("mac_app")


class InstallerTests(unittest.TestCase):
    def make_dmg(self, root: Path):
        """Run create_dmg with the OS utilities and Finder metadata stubbed."""
        staging, output = root / "staging", root / "output"
        staging.mkdir()
        output.mkdir()
        icns = root / "sigma.icns"
        icns.write_bytes(b"icns-artwork")
        dmg = output / "SIGMA.dmg"

        def fake_run(command, **kwargs):
            # Stand in for hdiutil attach, which is what puts the staged
            # artwork under the mount point the icon flag is applied to.
            if command[1] == "attach":
                mount = Path(command[command.index("-mountpoint") + 1])
                shutil.copy2(icns, mount / mac_icon.VOLUME_ICON_NAME)

        with patch.object(mac_app, "run", side_effect=fake_run) as run, \
                patch.object(mac_app.mac_icon, "mark_custom_icon") as mark, \
                patch.object(mac_app.mac_icon, "set_file_icon") as set_file_icon:
            mac_app.create_dmg(staging, dmg, icns)
        commands = [call.args[0] for call in run.call_args_list]
        return {"commands": commands, "staging": staging, "dmg": dmg, "icns": icns,
                "mark": mark, "set_file_icon": set_file_icon,
                "timeouts": [call.kwargs.get("timeout") for call in run.call_args_list]}

    def test_dmg_creation_has_diagnostics_and_a_bounded_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.make_dmg(Path(directory))
        create = next(c for c in result["commands"] if c[1] == "create")
        self.assertEqual(create[create.index("-fs") + 1], "APFS")
        self.assertIn("-verbose", create)
        self.assertIn("-nospotlight", create)
        self.assertTrue(all(timeout for timeout in result["timeouts"]))
        self.assertEqual(max(result["timeouts"]), 900)

    def test_dmg_is_written_read_write_then_compressed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.make_dmg(Path(directory))
        verbs = [command[1] for command in result["commands"]]
        self.assertEqual(verbs, ["create", "attach", "detach", "convert"])
        create = next(c for c in result["commands"] if c[1] == "create")
        convert = next(c for c in result["commands"] if c[1] == "convert")
        # The icon flag can only be set on a writable, attached image, so the
        # delivered file must still be produced by the compressing convert.
        self.assertEqual(create[create.index("-format") + 1], "UDRW")
        self.assertEqual(convert[convert.index("-format") + 1], "UDZO")
        self.assertEqual(Path(convert[convert.index("-o") + 1]), result["dmg"])

    def test_dmg_carries_volume_and_file_icons(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.make_dmg(Path(directory))
            staged = result["staging"] / mac_icon.VOLUME_ICON_NAME
            self.assertTrue(staged.is_file())
            self.assertEqual(staged.read_bytes(), result["icns"].read_bytes())
        # The volume root is flagged so Finder reads .VolumeIcon.icns, and the
        # finished image gets the same artwork in its resource fork.
        self.assertEqual(result["mark"].call_count, 1)
        result["set_file_icon"].assert_called_once_with(result["dmg"], result["icns"])

    def test_custom_icon_resource_fork_is_well_formed(self):
        artwork = b"icns" + (28).to_bytes(4, "big") + b"payload-bytes-here!!"
        fork = mac_icon.resource_fork(artwork)
        data_offset, map_offset, data_length, map_length = struct.unpack(">IIII", fork[:16])
        self.assertEqual(data_offset, 256)
        self.assertEqual(map_offset, 256 + data_length)
        self.assertEqual(len(fork), map_offset + map_length)
        resource_map = fork[map_offset:map_offset + map_length]
        type_list_offset, name_list_offset = struct.unpack(">HH", resource_map[24:28])
        # An empty name list sits at the very end of the map.
        self.assertEqual(name_list_offset, map_length)
        type_list = resource_map[type_list_offset:]
        self.assertEqual(struct.unpack(">H", type_list[:2])[0], 0)
        self.assertEqual(type_list[2:6], b"icns")
        count, reference_offset = struct.unpack(">HH", type_list[6:10])
        self.assertEqual(count, 0)
        reference = resource_map[type_list_offset + reference_offset:][:12]
        resource_id, name_offset = struct.unpack(">hh", reference[:4])
        self.assertEqual(resource_id, mac_icon.CUSTOM_ICON_RESOURCE_ID)
        self.assertEqual(name_offset, -1)
        start = data_offset + int.from_bytes(reference[5:8], "big")
        length = struct.unpack(">I", fork[start:start + 4])[0]
        self.assertEqual(fork[start + 4:start + 4 + length], artwork)

    def test_custom_icon_finder_flag_is_the_only_bit_set(self):
        info = mac_icon.finder_info()
        self.assertEqual(len(info), 32)
        self.assertEqual(struct.unpack(">H", info[8:10])[0], mac_icon.HAS_CUSTOM_ICON)
        self.assertEqual(info[:8], b"\0" * 8)
        self.assertEqual(info[10:], b"\0" * 22)

    def test_portable_python_archives_are_pinned_per_architecture(self):
        arm = mac_app.runtime_record("osx-arm64")
        intel = mac_app.runtime_record("osx-64")
        self.assertEqual(arm["python"], "3.13.15")
        self.assertEqual(intel["python"], "3.11.16")
        for record in (arm, intel):
            self.assertTrue(record["url"].startswith("https://github.com/astral-sh/python-build-standalone/releases/download/20260901/"))
            self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
            self.assertNotIn("freethreaded", record["filename"])

    def test_changed_python_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "runtime.tar.zst"
            archive.write_bytes(b"runtime")
            mac_app.check_digest(archive, hashlib.sha256(b"runtime").hexdigest())
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                mac_app.check_digest(archive, "0" * 64)

    @unittest.skipIf(sys.platform == "win32", "macOS symlink layout")
    def test_bundle_cannot_link_to_external_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "python3").write_bytes(b"python")
            (root / "python").symlink_to("python3")
            mac_app.check_internal_links(root)
            (root / "external").symlink_to("/Library/sigma-0.0.5")
            with self.assertRaisesRegex(ValueError, "external or broken"):
                mac_app.check_internal_links(root)

    def test_native_bundle_name_has_no_version_or_absolute_executable(self):
        plist = plistlib.loads(plistlib.dumps(mac_app.app_plist("0.0.5")))
        self.assertEqual(plist["CFBundleName"], "SIGMA")
        self.assertEqual(plist["CFBundleDisplayName"], "SIGMA")
        self.assertEqual(plist["CFBundleExecutable"], "SIGMA")
        self.assertEqual(plist["CFBundleVersion"], "0.0.5")

    def test_platform_versions(self):
        intel = build.requirements("osx-64")
        self.assertIn("numpy==1.26.4", intel)
        self.assertIn("torch==2.2.2", intel)
        self.assertIn("tifffile==2026.3.3", intel)
        self.assertIn("torch==2.13.0+cu130", build.requirements("win-64"))
        self.assertIn("torch==2.13.0", build.requirements("osx-arm64"))
        self.assertTrue(all("matplotlib==3.11.1" in build.requirements(target) for target in build.PLATFORMS))

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
        default_source = ROOT.parent if (ROOT.parent / "tests/_fixtures.py").exists() else ROOT.parents[1] / "napari-sigma"
        source = Path(os.environ.get("SIGMA_SOURCE_DIR", default_source))
        spec = importlib.util.spec_from_file_location("fixtures", source / "tests/_fixtures.py")
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
                if target != "win-64":
                    self.assertIn("unsigned", config["installer_filename"])
                self.assertEqual(list(config["extra_files"][0].values()), ["sigma-desktop/payload.txt"] if os.name != "nt" else ["sigma-desktop\\payload.txt"])
                if target == "win-64":
                    self.assertFalse(config["register_python"])
                    self.assertEqual(config["uninstall_name"], "SIGMA")
                else:
                    self.assertEqual(config["pkg_name"], f"sigma-{build.VERSION}")

    def test_icons_use_supplied_artwork_in_all_native_formats(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            build.make_icons(destination)
            source = ROOT / "assets/sigma-logo.png"
            self.assertEqual((destination / "sigma.png").read_bytes(), source.read_bytes())
            with Image.open(source) as original:
                self.assertEqual(original.mode, "RGBA")
                self.assertEqual(original.getchannel("A").getextrema(), (0, 255))
                self.assertEqual(original.getpixel((0, 0))[3], 0)
                self.assertEqual(original.getpixel((original.width // 2, original.height - 1))[3], 0)
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

    def test_windows_cuda_payload_is_not_embedded_in_small_exe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wheelhouse").mkdir()
            (root / "wheelhouse/torch.whl").write_bytes(b"cuda")
            (root / "bundle.json").write_text("{}")
            config = build.constructor_config("win-64", root / "runtime", root)
            included = [path for record in config["extra_files"] for path in record]
            self.assertEqual(included, [str(root / "bundle.json")])
            self.assertEqual(config["installer_filename"], "SIGMA-Setup.exe")

    def test_windows_zip_contains_setup_and_offline_wheels_with_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheels = root / "wheelhouse"
            wheels.mkdir()
            (wheels / "torch.whl").write_bytes(b"cuda")
            (root / "SIGMA-Setup.exe").write_bytes(b"setup")
            (root / "QUICKSTART.txt").write_text("Extract All")
            download = build.package_windows(root, wheels)
            with zipfile.ZipFile(download) as archive:
                self.assertEqual(set(archive.namelist()), {"SIGMA-Setup.exe", "QUICKSTART.txt", "wheelhouse/torch.whl"})
                self.assertEqual(archive.read("wheelhouse/torch.whl"), b"cuda")
            self.assertIn(hashlib.sha256(download.read_bytes()).hexdigest(),
                          download.with_suffix(".zip.sha256").read_text())

    def test_external_payload_is_resolved_beside_installer_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheels = root / "wheelhouse"
            wheels.mkdir()
            wheel = wheels / "torch.whl"
            wheel.write_bytes(b"cuda")
            bundle = {"wheelhouse_location": "next-to-installer", "wheels": [
                {"filename": wheel.name, "sha256": hashlib.sha256(b"cuda").hexdigest()}]}
            with patch.dict(os.environ, {"INSTALLER_PATH": str(root / "SIGMA-Setup.exe")}):
                self.assertEqual(install.validated_wheelhouse(root / "runtime/sigma-desktop", bundle), wheels.resolve())
                wheel.write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    install.validated_wheelhouse(root, bundle)
                wheel.unlink()
                with self.assertRaisesRegex(RuntimeError, "Extract All"):
                    install.validated_wheelhouse(root, bundle)
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "Extract All"):
                    install.validated_wheelhouse(root, bundle)

    def test_payload_cannot_escape_wheelhouse(self):
        for filename in ("../outside.whl", "..\\outside.whl", "/outside.whl"):
            bundle = {"wheels": [{"filename": filename, "sha256": "0" * 64}]}
            with self.assertRaisesRegex(ValueError, "Invalid wheel filename"):
                install.validated_wheelhouse(ROOT, bundle)

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
                self.assertEqual(os.environ["MPL_IGNORE_SYSTEM_FONTS"], "1")


if __name__ == "__main__":
    unittest.main()
