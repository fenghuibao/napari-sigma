"""No-console desktop entry point for the unchanged, published SIGMA core."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

RESOURCES = Path(__file__).resolve().parent


def user_directories() -> tuple[Path, Path]:
    if sys.platform == "darwin":
        return Path.home() / "Library/Logs/SIGMA", Path.home() / "Library/Caches/SIGMA"
    local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "SIGMA"
    return local / "logs", local / "cache"


def user_configuration() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/SIGMA"
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "SIGMA"


def prepare_process() -> Path:
    # Shortcuts use -I too, so inherited PYTHONPATH/user-site packages cannot
    # contaminate this installation. Keep all caches outside the private runtime.
    logs, cache = user_directories()
    configuration = user_configuration()
    for directory in (logs, cache / "numba", cache / "matplotlib", configuration):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache / "numba")
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["NAPARI_CONFIG"] = str(configuration / "napari.yaml")
    return logs / "desktop.log"


def smoke_checks(viewer, panel) -> dict:
    import numpy as np
    import napari_sigma
    import torch
    from napari_sigma._writer import write_single_labels
    from napari_sigma.segmentation import segmentation
    from frangi_filter.frangi_filter import FrangiFilter

    with tempfile.TemporaryDirectory(prefix="sigma-smoke-") as directory:
        path = str(Path(directory) / "校准 labels.tif")
        labels = np.full((2, 3, 8, 9), 2**40 + 1, np.uint64)
        write_single_labels(path, labels, {"scale": (1, 2, .3, .2), "metadata": {"dims": "TZYX"}})
        layer = viewer.open(path, plugin="napari-sigma")[0]
        np.testing.assert_array_equal(layer.data, labels)
        np.testing.assert_allclose(layer.scale, (1, 2, .3, .2))
    raw = np.arange(256, dtype=np.uint16).reshape(16, 16)
    response = raw.astype(np.float32) / 255
    result, _ = segmentation(raw, response, pixel_size=(.1, .1), beta1=.5, beta2=1.,
                             n_fore=1, n_back=1, max_iter=2, init_method="otsu")
    # Core segmentation retains a leading singleton axis for 2-D input.
    assert result.shape == (1, *raw.shape)
    assert result.dtype == np.uint8 and np.isin(result, (0, 255)).all()
    model = FrangiFilter(1, 5, [1.], 2, psf_ratio=1.)
    filtered = model(-torch.from_numpy(raw.astype(np.float32))[None, None])
    assert bool(torch.isfinite(filtered).all())
    return {"status": "ok", "sigma": napari_sigma.__version__, "torch": torch.__version__,
            "python": sys.version.split()[0], "panel": type(panel).__name__,
            "reader": "TIFF uint64 TZYX calibrated", "frangi": "ok", "segmentation": "ok"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args(argv)
    log_path = None
    splash = panel = viewer = None
    try:
        log_path = prepare_process()
        # pythonw has no console streams. Native/library error messages still
        # need valid streams, and a readable log is preferable to a silent exit.
        if not args.smoke_test:
            log_stream = log_path.open("a", encoding="utf-8", buffering=1)
            sys.stdout = sys.stderr = log_stream
        from napari_sigma._launcher import _configure_pyqt6
        _configure_pyqt6()
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QIcon, QPixmap, QPainter, QColor, QFont
        from qtpy.QtWidgets import QApplication, QSplashScreen
        app = QApplication.instance() or QApplication(["SIGMA"])
        app.setApplicationName("SIGMA")
        app.setWindowIcon(QIcon(str(RESOURCES / "sigma.png")))
        pixmap = QPixmap(480, 170)
        pixmap.fill(QColor("#182838"))
        painter = QPainter(pixmap)
        painter.setPen(QColor("#72e0bd"))
        painter.setFont(QFont("Arial", 28, QFont.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "SIGMA")
        painter.end()
        splash = QSplashScreen(pixmap)
        splash.showMessage("Loading microscopy tools…", Qt.AlignBottom | Qt.AlignHCenter, QColor("white"))
        splash.show()
        app.processEvents()

        import napari
        import napari_sigma
        from napari_sigma._widget import SIGMAWidget
        bundle = json.loads((RESOURCES / "bundle.json").read_text(encoding="utf-8"))
        if napari_sigma.__version__ != bundle["sigma_version"]:
            raise RuntimeError("SIGMA installation version mismatch. Reinstall the complete desktop package.")
        viewer = napari.Viewer(title=f"SIGMA {napari_sigma.__version__}", show=False)
        panel = SIGMAWidget(viewer)
        dock = viewer.window.add_dock_widget(panel, name="SIGMA")
        panel.device_combo.setCurrentText(bundle["default_device"])
        viewer.show()
        # The scientific panel has a substantial minimum width. A small default
        # napari window can otherwise leave no visible image canvas.
        viewer.window._qt_window.showMaximized()
        viewer.window._qt_window.resizeDocks([dock], [panel.minimumWidth()], Qt.Horizontal)
        splash.finish(viewer.window._qt_window)
        app.processEvents()
        if args.smoke_test:
            print(json.dumps(smoke_checks(viewer, panel)), flush=True)
            app.processEvents()
            if args.screenshot:
                if not viewer.window._qt_window.grab().save(str(args.screenshot)):
                    raise RuntimeError("Could not save smoke-test screenshot.")
            return 0
        napari.run()
        return 0
    except Exception:
        details = traceback.format_exc()
        if log_path:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(details + "\n")
        if args.smoke_test:
            print(details, file=sys.stderr)
            return 1
        try:
            from qtpy.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance() or QApplication(["SIGMA"])
            if splash:
                splash.close()
            box = QMessageBox()
            box.setWindowTitle("SIGMA could not start")
            box.setIcon(QMessageBox.Critical)
            box.setText("SIGMA could not start. Please keep the details below for support.")
            box.setInformativeText(f"Log: {log_path or 'unavailable'}")
            box.setDetailedText(details)
            box.exec()
        except Exception:
            # Even a Qt load failure must be visible without a terminal.
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, f"SIGMA could not start.\nLog: {log_path}\n{details}", "SIGMA", 0x10)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.run(["/usr/bin/osascript", "-e",
                                'on run argv\ndisplay alert "SIGMA could not start" message (item 1 of argv) as critical\nend run',
                                f"Log: {log_path}\n{details}"], check=False)
        return 1
    finally:
        if args.smoke_test:
            if panel is not None:
                panel.dispose()
            if viewer is not None:
                viewer.close()
            if splash is not None:
                splash.close()


if __name__ == "__main__":
    raise SystemExit(main())
