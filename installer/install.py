"""Compulsory offline installer step. Never modifies a user's other Python."""
from __future__ import annotations

import argparse
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
    log = resources / "installation.log"
    with log.open("a", encoding="utf-8") as stream:
        subprocess.run([
            sys.executable, "-I", "-m", "pip", "install", "--no-index",
            "--find-links", str(resources / "wheelhouse"), "--require-hashes", "--no-deps",
            "--no-cache-dir", "--no-compile", "--disable-pip-version-check", "--ignore-installed",
            "--root-user-action=ignore", "-r", str(resources / "requirements.lock"),
        ], check=True, stdout=stream, stderr=subprocess.STDOUT)
        subprocess.run([sys.executable, "-I", "-m", "pip", "check"],
                       check=True, stdout=stream, stderr=subprocess.STDOUT)
    bundle = json.loads((resources / "bundle.json").read_text(encoding="utf-8"))
    subprocess.run([sys.executable, "-I", "-c",
                    "import napari_sigma; assert napari_sigma.__version__ == " + repr(bundle["sigma_version"])], check=True)
    if not args.no_shortcuts:
        paths = install(resources / "menu.json", target_prefix=str(prefix), base_prefix=str(prefix), _mode=mode)
        (resources / "shortcuts.json").write_text(json.dumps({"mode": mode, "paths": [str(p) for p in paths]}), encoding="utf-8")
    print("SIGMA runtime and shortcuts are ready.", flush=True)


if __name__ == "__main__":
    main()
