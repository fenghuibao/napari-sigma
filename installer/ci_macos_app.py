"""Install, relocate, verify and delete a DMG's app on a disposable Mac runner."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import mac_icon
from verify import run_native_smoke


def check_volume_icon(mount: Path):
    """Finder needs both the artwork and the flag to draw a volume icon."""
    artwork = mount / mac_icon.VOLUME_ICON_NAME
    if not artwork.is_file() or not artwork.stat().st_size:
        raise RuntimeError(f"The disk image has no volume icon artwork: {artwork}")
    info = subprocess.run(["/usr/bin/xattr", "-px", "com.apple.FinderInfo", str(mount)],
                          check=True, capture_output=True, text=True).stdout
    flags = int.from_bytes(bytes.fromhex(info.replace("\n", "").replace(" ", ""))[8:10], "big")
    if not flags & mac_icon.HAS_CUSTOM_ICON:
        raise RuntimeError(f"The disk image volume is not flagged to show its icon: {flags:#06x}")
    print(f"Volume icon present: {artwork.stat().st_size} bytes, Finder flags {flags:#06x}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dmg", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Preserve real installations; only require that this build creates no new
    # external runtime. Native startup from the read-only DMG and signature
    # verification independently exercise the self-contained app.
    libraries = [Path("/Library"), Path.home() / "Library"]
    external_before = {path for root in libraries for path in root.glob("sigma-*")}
    build = args.build_dir.resolve()
    hidden = build.with_name(build.name + "-hidden-for-relocation-test")
    if hidden.exists():
        raise FileExistsError(hidden)
    build.rename(hidden)
    try:
        with tempfile.TemporaryDirectory(prefix="sigma-dmg-test-") as directory:
            root = Path(directory)
            mount = root / "mounted"
            subprocess.run(["/usr/bin/hdiutil", "attach", str(args.dmg.resolve()), "-readonly",
                            "-nobrowse", "-mountpoint", str(mount)], check=True)
            try:
                app = mount / "SIGMA.app"
                check_volume_icon(mount)
                # Native startup from a read-only volume cannot repair paths or
                # unpack a hidden runtime. It must work exactly as shipped.
                run_native_smoke(app, output / "read-only", os.environ.copy())
                copied = root / "first location/SIGMA.app"
                copied.parent.mkdir()
                subprocess.run(["/usr/bin/ditto", str(app), str(copied)], check=True)
            finally:
                subprocess.run(["/usr/bin/hdiutil", "detach", str(mount)], check=True)
            # Spaces and non-ASCII characters, without baking a build-machine
            # path in another language into the shipped verification logs.
            relocated = root / "Relocated Applications (café)/SIGMA.app"
            relocated.parent.mkdir()
            copied.rename(relocated)
            subprocess.run([sys.executable, str(Path(__file__).with_name("verify.py")),
                            "--app", str(relocated), "--output-dir", str(output),
                            "--source-dir", str(args.source_dir.resolve())], check=True)
            # This is only the test-owned app in TemporaryDirectory, never a
            # real user's installation, data or cache directory.
            shutil.rmtree(relocated)
            assert not relocated.exists()
            external_after = {path for base in libraries for path in base.glob("sigma-*")}
            assert external_after == external_before
        (output / "relocation.json").write_text(json.dumps({
            "status": "passed", "read_only_native_launch": True, "volume_icon": True,
            "build_location_unavailable": True, "relocated_path_with_spaces_and_unicode": True,
            "app_unchanged_after_tests": True, "delete_app_removes_runtime": True,
            "no_external_runtime_created": True,
        }, indent=2), encoding="utf-8")
    finally:
        hidden.rename(build)


if __name__ == "__main__":
    main()
