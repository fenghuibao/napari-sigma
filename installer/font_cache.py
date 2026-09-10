"""Restore SIGMA's version-matched bundled font index before font-manager import.

No system font discovery, global Matplotlib patches, or machine-specific font
paths. Corrupt user caches are atomically replaced; damaged/mismatched installed
assets raise an explicit error instead of silently scanning the operating system.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def prepare_font_cache(resources: Path) -> Path:
    # Official Matplotlib switch: even a later cache rebuild stays app-local.
    os.environ["MPL_IGNORE_SYSTEM_FONTS"] = "1"
    # Importing matplotlib itself does not initialize its font manager.
    import matplotlib as mpl

    source = resources / "font-cache"
    manifest_bytes = (source / "manifest.json").read_bytes()
    bundle = json.loads((resources / "bundle.json").read_text(encoding="utf-8"))
    if hashlib.sha256(manifest_bytes).hexdigest() != bundle["font_cache"]["manifest_sha256"]:
        raise RuntimeError("Bundled font manifest is damaged. Reinstall SIGMA.")
    manifest = json.loads(manifest_bytes)
    if manifest["schema"] != 1 or manifest["matplotlib_version"] != mpl.__version__:
        raise RuntimeError("Bundled fonts do not match Matplotlib. Reinstall the complete SIGMA app.")
    name = manifest["cache_filename"]
    if not re.fullmatch(r"fontlist-v[0-9A-Za-z_.-]+\.json", name):
        raise RuntimeError("Invalid bundled font index filename")
    content = (source / name).read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest["cache_sha256"]:
        raise RuntimeError("Bundled font index is damaged. Reinstall SIGMA.")
    index = json.loads(content)
    if name != f"fontlist-v{index['_version']}.json":
        raise RuntimeError("Bundled font cache schema mismatch")
    data = Path(mpl.get_data_path()).resolve()
    fonts = manifest["fonts"]
    indexed_paths = {entry["fname"] for kind in ("ttflist", "afmlist") for entry in index[kind]}
    if not fonts or indexed_paths != {font["path"] for font in fonts}:
        raise RuntimeError("Bundled font manifest does not cover the index")
    for font in fonts:
        relative = Path(font["path"])
        path = (data / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(data / "fonts"):
            raise RuntimeError("Bundled font path escapes the application")
        if hashlib.sha256(path.read_bytes()).hexdigest() != font["sha256"]:
            raise RuntimeError(f"Bundled font is missing or damaged: {relative}. Reinstall SIGMA.")
    cache_dir = Path(mpl.get_cachedir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / name
    try:
        current = target.read_bytes()
    except FileNotFoundError:
        current = None
    if current != content:
        # Multiple app instances can safely restore the same immutable bytes.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache_dir, prefix="sigma-fonts-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return target
