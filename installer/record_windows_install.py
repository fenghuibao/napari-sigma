"""Record only the shortcuts created by this installer, without importing Qt."""
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent
    if root != Path(sys.prefix).resolve() / "sigma-desktop":
        raise RuntimeError("Run this helper with SIGMA's private Python runtime")
    paths = [Path(p).resolve() for p in sys.argv[1:]]
    if len(paths) != 2 or any(p.name != "SIGMA.lnk" or not p.is_file() for p in paths):
        raise RuntimeError("Both SIGMA shortcuts must exist")
    (root / "shortcuts.json").write_text(json.dumps({"paths": [str(p) for p in paths]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
