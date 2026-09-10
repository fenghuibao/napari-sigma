"""Build a relocatable index of only the fonts shipped in the locked wheel.

Run with the target Matplotlib in a disposable build environment, never with
the user's app. No system font is allowed into the distributable index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def generate(destination: Path):
    os.environ["MPL_IGNORE_SYSTEM_FONTS"] = "1"
    import matplotlib as mpl
    from matplotlib import font_manager as fm

    data = Path(mpl.get_data_path()).resolve()
    manager = fm.fontManager
    for attr in ("ttflist", "afmlist"):
        entries = [entry for entry in getattr(manager, attr)
                   if Path(entry.fname).resolve().is_relative_to(data / "fonts")]
        if not entries:
            raise RuntimeError(f"No bundled fonts found in {attr}")
        setattr(manager, attr, entries)
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination / f"fontlist-v{fm.FontManager.__version__}.json"
    fm.json_dump(manager, cache)
    index = json.loads(cache.read_text(encoding="utf-8"))
    paths = sorted({entry["fname"] for kind in ("ttflist", "afmlist") for entry in index[kind]})
    fonts = []
    for path in paths:
        relative = Path(path)
        if relative.is_absolute() or not (data / relative).resolve().is_relative_to(data / "fonts"):
            raise RuntimeError(f"Font index contains a non-portable path: {path}")
        fonts.append({"path": path, "sha256": hashlib.sha256((data / path).read_bytes()).hexdigest()})
    manifest = {"schema": 1, "matplotlib_version": mpl.__version__,
                "cache_filename": cache.name, "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                "fonts": fonts}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Indexed {len(fonts)} bundled font files for Matplotlib {mpl.__version__}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    generate(parser.parse_args().output)
