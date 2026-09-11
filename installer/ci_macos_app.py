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

from verify import run_native_smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dmg", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    external = [Path("/Library/sigma-0.0.5"), Path.home() / "Library/sigma-0.0.5"]
    if any(path.exists() for path in external):
        raise RuntimeError("This test requires a disposable Mac without an older SIGMA runtime")
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
                # Native startup from a read-only volume cannot repair paths or
                # unpack a hidden runtime. It must work exactly as shipped.
                run_native_smoke(app, output / "read-only", os.environ.copy())
                copied = root / "first location/SIGMA.app"
                copied.parent.mkdir()
                subprocess.run(["/usr/bin/ditto", str(app), str(copied)], check=True)
            finally:
                subprocess.run(["/usr/bin/hdiutil", "detach", str(mount)], check=True)
            relocated = root / "搬移后 Applications/SIGMA.app"
            relocated.parent.mkdir()
            copied.rename(relocated)
            subprocess.run([sys.executable, str(Path(__file__).with_name("verify.py")),
                            "--app", str(relocated), "--output-dir", str(output)], check=True)
            # This is only the test-owned app in TemporaryDirectory, never a
            # real user's installation, data or cache directory.
            shutil.rmtree(relocated)
            assert not relocated.exists()
            assert not any(path.exists() for path in external)
        (output / "relocation.json").write_text(json.dumps({
            "status": "passed", "read_only_native_launch": True,
            "build_location_unavailable": True, "relocated_path_with_spaces_and_unicode": True,
            "app_unchanged_after_tests": True, "delete_app_removes_runtime": True,
            "no_external_runtime_created": True,
        }, indent=2), encoding="utf-8")
    finally:
        hidden.rename(build)


if __name__ == "__main__":
    main()
