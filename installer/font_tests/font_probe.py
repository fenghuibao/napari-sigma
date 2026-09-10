"""Fresh-process checks of real font imports/rendering and the desktop entrypoint."""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import threading
import time


@contextmanager
def no_font_discovery():
    calls = []
    def record(frame, event, arg):
        if event == "call" and frame.f_code.co_name in {
                "findSystemFonts", "_get_fontconfig_fonts", "_get_macos_fonts", "_get_win32_installed_fonts"}:
            if frame.f_code.co_filename.replace("\\", "/").endswith("/matplotlib/font_manager.py"):
                calls.append(frame.f_code.co_name)
    sys.setprofile(record)
    threading.setprofile(record)
    try:
        yield calls
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
        assert not calls, f"Unexpected font discovery: {calls}"


def probe(resources: Path, scenario: str, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(output / "cache")
    spec = importlib.util.spec_from_file_location("sigma_fonts_probe", resources / "font_cache.py")
    fonts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fonts)
    with no_font_discovery():
        import matplotlib as mpl
        assert "matplotlib.font_manager" not in sys.modules
        if scenario in {"relocated", "damaged-font"}:
            data = output / "relocated application" / "mpl-data"
            shutil.copytree(mpl.get_data_path(), data)
            mpl.get_data_path = lambda: str(data)
            if scenario == "damaged-font":
                (data / "fonts/ttf/DejaVuSans.ttf").write_bytes(b"invalid font")
        if scenario == "wrong-version":
            mpl.__version__ = "0.0.invalid"
        if scenario == "damaged-index":
            source = output / "damaged resources"
            shutil.copytree(resources / "font-cache", source / "font-cache")
            shutil.copy2(resources / "bundle.json", source / "bundle.json")
            manifest = json.loads((source / "font-cache/manifest.json").read_text())
            (source / "font-cache" / manifest["cache_filename"]).write_bytes(b"bad index")
            resources = source
        start = time.perf_counter()
        if scenario in {"wrong-version", "damaged-index", "damaged-font"}:
            try:
                fonts.prepare_font_cache(resources)
            except RuntimeError as exc:
                assert "Reinstall" in str(exc), exc
                assert "matplotlib.font_manager" not in sys.modules
                print(json.dumps({"scenario": scenario, "status": "safely rejected", "error": str(exc)}))
                return
            raise AssertionError("Damaged/mismatched installation was accepted")
        cache = fonts.prepare_font_cache(resources)
        expected = cache.read_bytes()
        if scenario == "cache-recovery":
            cache.write_bytes(b"broken user cache")
            fonts.prepare_font_cache(resources)
            assert cache.read_bytes() == expected
            cache.unlink()
            fonts.prepare_font_cache(resources)
            assert cache.read_bytes() == expected
        restore_seconds = time.perf_counter() - start
        from matplotlib import font_manager as fm
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        assert fm.FontManager.__version__ == fm.fontManager._version
        root = Path(mpl.get_data_path()).resolve() / "fonts"
        assert all(Path(entry.fname).resolve().is_relative_to(root)
                   for entry in fm.fontManager.ttflist + fm.fontManager.afmlist)
        figure = Figure()
        canvas = FigureCanvasAgg(figure)
        axes = figure.subplots()
        axes.plot([0, 1, 2], [0, 1, .5], label="SIGMA")
        axes.set_title("Morphology: σ, μm, −1; $x^2$")
        axes.set_xlabel("Length (μm)")
        axes.legend()
        canvas.print_png(output / "fonts.png")
        figure.savefig(output / "fonts.svg")
        figure.savefig(output / "fonts.pdf")
        assert cache.read_bytes() == expected, "Matplotlib rebuilt the pre-generated index"
        print(json.dumps({"scenario": scenario, "status": "passed", "font_discovery_calls": 0,
                          "restore_seconds": restore_seconds,
                          "restore_import_render_seconds": time.perf_counter() - start,
                          "font_files": len({entry.fname for entry in fm.fontManager.ttflist + fm.fontManager.afmlist})}))


if __name__ == "__main__":
    if sys.argv[1] == "--launch":
        script = sys.argv[2]
        sys.argv = sys.argv[2:]
        with no_font_discovery():
            runpy.run_path(script, run_name="__main__")
    else:
        probe(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
