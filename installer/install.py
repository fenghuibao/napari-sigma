"""Compulsory offline installer step. Never modifies a user's other Python."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def install_prefix() -> Path:
    prefix = Path(sys.prefix).resolve()
    if Path(__file__).resolve().parent != prefix / "sigma-desktop":
        raise RuntimeError("Installer must run with its own bundled Python.")
    return prefix


def menu_mode(prefix: Path) -> str:
    if os.name == "nt":
        return "user" if (prefix / ".nonadmin").exists() else "system"
    return "system" if os.geteuid() == 0 else "user"


def pip_environment() -> dict[str, str]:
    # User pip settings such as PIP_TARGET/PIP_PREFIX must never redirect writes
    # out of the bundled runtime, nor enable an external package index.
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("PIP_")}
    env["PIP_CONFIG_FILE"] = os.devnull
    return env


def validated_wheelhouse(resources: Path, bundle: dict) -> Path:
    wheels = resources / "wheelhouse"
    if bundle.get("wheelhouse_location") == "next-to-installer":
        # Constructor 3.16.1 sets INSTALLER_PATH from NSIS $EXEPATH, replacing
        # any inherited value. No shell expansion or network source is used.
        installer = os.environ.get("INSTALLER_PATH")
        if not installer:
            raise RuntimeError("Use Extract All on the download, then run SIGMA-Setup.exe.")
        wheels = Path(installer).resolve().parent / "wheelhouse"
    for record in bundle["wheels"]:
        name = record["filename"]
        if Path(name).name != name or "/" in name or "\\" in name or not name.endswith(".whl"):
            raise ValueError("Invalid wheel filename in bundle")
        wheel = wheels / name
        if not wheel.is_file():
            raise RuntimeError(f"Missing {name}. Use Extract All, then run SIGMA-Setup.exe "
                               "beside the complete wheelhouse folder.")
        with wheel.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {name}. Download the complete installer again.")
    return wheels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove-shortcuts", action="store_true")
    parser.add_argument("--no-shortcuts", action="store_true")
    args = parser.parse_args()
    prefix = install_prefix()
    resources = prefix / "sigma-desktop"
    from menuinst.api import install, remove
    mode = menu_mode(prefix)
    if args.remove_shortcuts:
        remove(resources / "menu.json", target_prefix=str(prefix), base_prefix=str(prefix), _mode=mode)
        return
    bundle = json.loads((resources / "bundle.json").read_text(encoding="utf-8"))
    wheels = validated_wheelhouse(resources, bundle)
    log = resources / "installation.log"
    env = pip_environment()
    with log.open("a", encoding="utf-8") as stream:
        subprocess.run([
            sys.executable, "-I", "-m", "pip", "--isolated", "install", "--no-index",
            "--find-links", str(wheels), "--require-hashes", "--no-deps",
            "--no-cache-dir", "--no-compile", "--disable-pip-version-check", "--ignore-installed",
            "--root-user-action=ignore", "-r", str(resources / "requirements.lock"),
        ], check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
        subprocess.run([sys.executable, "-I", "-m", "pip", "--isolated", "check"],
                       check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
    subprocess.run([sys.executable, "-I", "-c",
                    "import napari_sigma; assert napari_sigma.__version__ == " + repr(bundle["sigma_version"])], check=True)
    if not args.no_shortcuts:
        paths = install(resources / "menu.json", target_prefix=str(prefix), base_prefix=str(prefix), _mode=mode)
        (resources / "shortcuts.json").write_text(json.dumps({"mode": mode, "paths": [str(p) for p in paths]}), encoding="utf-8")
    print("SIGMA runtime and shortcuts are ready.", flush=True)


if __name__ == "__main__":
    main()
