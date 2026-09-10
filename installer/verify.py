"""Test an installed distribution, not the source checkout/build environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile


def run_smoke(python: Path, resources: Path, output: Path, env: dict):
    # The child keeps TIFF memory maps alive until GUI shutdown. In particular,
    # Windows cannot unlink them earlier. Cleanup errors must not be suppressed.
    with tempfile.TemporaryDirectory(prefix="sigma-smoke-") as directory:
        command = [str(python), "-I", str(resources / "launch.py"), "--smoke-test",
                   "--smoke-data-dir", directory, "--screenshot", str(output / "desktop.png")]
        return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              env=env, timeout=1200)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    resources = prefix / "sigma-desktop"
    bundle = json.loads((resources / "bundle.json").read_text(encoding="utf-8"))
    source_logo = Path(__file__).resolve().parent / "assets/sigma-logo.png"
    if (resources / "sigma.png").read_bytes() != source_logo.read_bytes():
        raise RuntimeError("Installed logo differs from the supplied artwork")
    assert bundle["branding"]["logo_sha256"] == hashlib.sha256(source_logo.read_bytes()).hexdigest()
    python = prefix / ("python.exe" if sys.platform == "win32" else "bin/python")
    env = os.environ.copy()
    env["SIGMA_DESKTOP_TEST_RESOURCES"] = str(resources)
    env.update(NUMBA_CACHE_DIR=str(output / "numba-cache"), MPLCONFIGDIR=str(output / "mpl-cache"), PYTHONWARNINGS="ignore")
    # A poisoned PYTHONPATH must not affect the -I entry point.
    poison = output / "poison-path"
    poison.mkdir(exist_ok=True)
    (poison / "napari_sigma.py").write_text("raise RuntimeError('External PYTHONPATH was loaded')\n", encoding="utf-8")
    env["PYTHONPATH"] = str(poison)
    commands = [
        [python, "-I", "-c", "import sys, pathlib, napari_sigma; p=pathlib.Path(napari_sigma.__file__).resolve(); assert p.is_relative_to(pathlib.Path(sys.prefix).resolve()), p; print(p)"],
        [python, "-I", "-m", "pip", "check"],
        [python, "-I", resources / "launch.py", "--smoke-test", "--screenshot", output / "desktop.png"],
        [python, "-I", "-m", "unittest", "discover", "-s", Path(__file__).resolve().parents[1] / "tests", "-v"],
        [python, "-I", "-m", "unittest", "discover", "-s", Path(__file__).resolve().parent / "gui_tests", "-v"],
    ]
    for index, command in enumerate(commands):
        command = [str(arg) for arg in command]
        print(subprocess.list2cmdline(command), flush=True)
        # The existing tests also create non-isolated child interpreters. Keep
        # those children pointed at the installed core, never the checkout.
        if index == 3:
            env.pop("PYTHONPATH", None)
        if index == 2:
            result = run_smoke(python, resources, output, env)
        else:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=1200)
        (output / f"check-{index}.log").write_bytes(result.stdout)
        print(result.stdout.decode("utf-8", errors="replace"), flush=True)
        result.check_returncode()
    shortcuts = json.loads((resources / "shortcuts.json").read_text(encoding="utf-8"))
    if not shortcuts["paths"]:
        raise RuntimeError("No desktop shortcut was installed")
    for path in shortcuts["paths"]:
        if not Path(path).exists():
            raise RuntimeError(f"Missing installed shortcut: {path}")
    if sys.platform == "darwin":
        apps = [Path(path) for path in shortcuts["paths"] if path.endswith(".app")]
        if len(apps) != 1 or apps[0].name != "SIGMA.app":
            raise RuntimeError(f"Expected an unversioned SIGMA.app: {apps}")
        with (apps[0] / "Contents/Info.plist").open("rb") as stream:
            plist = plistlib.load(stream)
        assert plist["CFBundleDisplayName"] == "SIGMA"
        assert plist["CFBundleName"] == "SIGMA"
        installed_icon = apps[0] / "Contents/Resources" / plist["CFBundleIconFile"]
        assert installed_icon.read_bytes() == (resources / "sigma.icns").read_bytes()
    elif sys.platform == "win32":
        links = [Path(path) for path in shortcuts["paths"] if path.endswith(".lnk")]
        if not links or any(path.name != "SIGMA.lnk" for path in links):
            raise RuntimeError(f"Expected unversioned SIGMA shortcuts: {links}")
    (output / "verification.json").write_text(json.dumps({
        "status": "passed", "prefix": str(prefix), "shortcuts": shortcuts,
        "bundle": bundle,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
