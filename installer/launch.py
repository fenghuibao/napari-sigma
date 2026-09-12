"""No-console desktop entry point for the unchanged, published SIGMA core."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
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
    os.environ["MPL_IGNORE_SYSTEM_FONTS"] = "1"
    os.environ["NAPARI_CONFIG"] = str(configuration / "napari.yaml")
    return logs / "desktop.log"


def smoke_checks(viewer, panel, directory: Path) -> dict:
    import numpy as np
    import napari_sigma
    import torch
    from napari_sigma._writer import write_single_labels
    from napari_sigma.segmentation import segmentation
    from frangi_filter.frangi_filter import FrangiFilter

    # verify.py owns this directory and removes it only after this GUI process
    # exits. napari keeps a memory map alive while displaying the TIFF, which
    # correctly prevents deletion of the backing file on Windows.
    # Keeps the space and non-ASCII coverage without writing a build-machine
    # filename in another language into the shipped verification logs.
    path = str(directory / "calibración labels.tif")
    labels = np.zeros((2, 3, 8, 9), np.uint64)
    labels[:, :, 1:7, 2:8] = 2**40 + 1
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
    parser.add_argument("--smoke-data-dir", type=Path)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args(argv)
    if args.smoke_test and (args.smoke_data_dir is None or not args.smoke_data_dir.is_dir()):
        parser.error("--smoke-test requires an existing --smoke-data-dir owned by the parent process")
    log_path = None
    splash = panel = viewer = None
    try:
        log_path = prepare_process()
        # pythonw has no console streams. Native/library error messages still
        # need valid streams, and a readable log is preferable to a silent exit.
        if not args.smoke_test:
            log_stream = log_path.open("a", encoding="utf-8", buffering=1)
            sys.stdout = sys.stderr = log_stream
        from napari_sigma._launcher import _configure_pyqt6, prefer_sigma_reader
        _configure_pyqt6()
        # A dropped file must arrive calibrated: napari draws inside
        # LayerList.insert, ahead of every plugin callback, so a layer that
        # starts on the default "pixel" unit makes napari report inconsistent
        # units before SIGMA can repair it. See prefer_sigma_reader.
        # force=True: this is SIGMA's own application and its own settings
        # directory (NAPARI_CONFIG above), not a shared napari where another
        # plugin's reader assignment deserves to be left alone.
        prefer_sigma_reader(force=True)
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QIcon, QPixmap, QPainter, QColor
        from qtpy.QtWidgets import QApplication, QSplashScreen
        app = QApplication.instance() or QApplication(["SIGMA"])
        app.setApplicationName("SIGMA")
        app.setApplicationDisplayName("SIGMA")
        icon = QIcon(str(RESOURCES / "sigma.png"))
        if icon.isNull():
            raise RuntimeError("The SIGMA app icon is missing or invalid")
        app.setWindowIcon(icon)
        pixel_ratio = app.primaryScreen().devicePixelRatio()
        pixmap = QPixmap(round(420 * pixel_ratio), round(440 * pixel_ratio))
        pixmap.setDevicePixelRatio(pixel_ratio)
        pixmap.fill(Qt.transparent)
        logo = QPixmap(str(RESOURCES / "sigma.png")).scaled(
            round(380 * pixel_ratio), round(380 * pixel_ratio),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo.setDevicePixelRatio(pixel_ratio)
        painter = QPainter(pixmap)
        painter.drawPixmap(20, 10, logo)
        painter.end()
        splash = QSplashScreen(pixmap)
        splash.setWindowFlag(Qt.FramelessWindowHint)
        splash.setAttribute(Qt.WA_TranslucentBackground)
        splash.setWindowTitle("SIGMA")
        splash.showMessage("Loading microscopy tools…", Qt.AlignBottom | Qt.AlignHCenter, QColor("#082b50"))
        splash.show()
        app.processEvents()
        if args.smoke_test and args.screenshot:
            if not splash.grab().save(str(args.screenshot.with_name("splash.png"))):
                raise RuntimeError("Could not save startup-screen screenshot")

        import napari
        import napari_sigma
        # -I deliberately excludes the script directory from sys.path. Load
        # only this explicitly trusted bundled desktop adapter by file path.
        import importlib.util
        spec = importlib.util.spec_from_file_location("sigma_desktop_widget", RESOURCES / "desktop_widget.py")
        desktop = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(desktop)
        bundle = json.loads((RESOURCES / "bundle.json").read_text(encoding="utf-8"))
        if napari_sigma.__version__ != bundle["sigma_version"]:
            raise RuntimeError("SIGMA installation version mismatch. Reinstall the complete desktop package.")
        # The window and dock title bars carry the expanded name; the
        # application, Dock tile and splash stay "SIGMA".
        viewer = napari.Viewer(title=desktop.FULL_NAME, show=False)
        # napari sets its own application icon during window creation/theme
        # changes. Restore SIGMA branding, including an explicit window icon.
        def restore_icon(*_):
            app.setWindowIcon(icon)
            viewer.window._qt_window.setWindowIcon(icon)
        viewer.events.theme.connect(restore_icon)
        restore_icon()
        panel = desktop.DesktopSIGMAWidget(viewer)
        dock = viewer.window.add_dock_widget(panel, name=desktop.FULL_NAME)
        # A no-op when this build's accelerator is not actually present: the
        # combo lists only usable devices (best first) and drops the rest.
        panel.device_combo.setCurrentText(bundle["default_device"])
        viewer.show()
        # The scientific panel has a substantial minimum width. A small default
        # napari window can otherwise leave no visible image canvas.
        viewer.window._qt_window.showMaximized()
        mac_window = None
        if sys.platform == "darwin":
            native_spec = importlib.util.spec_from_file_location("sigma_mac_window", RESOURCES / "mac_window.py")
            mac_window = importlib.util.module_from_spec(native_spec)
            native_spec.loader.exec_module(mac_window)
            mac_window.center_window_title(viewer.window._qt_window)
        # A 1024-wide display cannot fit both sidebars plus the image. Keep
        # layer controls/list accessible as tabs beside SIGMA on small screens.
        if viewer.window._qt_window.screen().availableGeometry().width() < 1280:
            for other in (viewer.window._qt_viewer.dockLayerControls,
                          viewer.window._qt_viewer.dockLayerList):
                viewer.window._qt_window.tabifyDockWidget(dock, other)
            dock.raise_()
        viewer.window._qt_window.resizeDocks([dock], [panel.minimumWidth()], Qt.Horizontal)
        splash.finish(viewer.window._qt_window)
        app.processEvents()
        if args.smoke_test:
            if mac_window is not None:
                import math
                offset = mac_window.title_center_offset(viewer.window._qt_window)
                if not math.isfinite(offset) or abs(offset) > 2:
                    raise RuntimeError(f"Native title is not centered: offset={offset} points")
                print(json.dumps({"native_title_center_offset_points": offset}), flush=True)
            window_title = viewer.window._qt_window.windowTitle()
            if dock.windowTitle() != desktop.FULL_NAME:
                raise RuntimeError(f"The panel title bar must show the full SIGMA name: {dock.windowTitle()!r}")
            # napari owns the window title; require the name, not sole ownership.
            if desktop.FULL_NAME not in window_title:
                raise RuntimeError(f"The window title bar must show the full SIGMA name: {window_title!r}")
            if app.applicationDisplayName() != "SIGMA":
                raise RuntimeError("The application name must stay SIGMA")
            if bundle["sigma_version"] in window_title or bundle["sigma_version"] in dock.windowTitle():
                raise RuntimeError("Titles must not carry a version suffix")
            if viewer.window._qt_window.windowIcon().cacheKey() != icon.cacheKey():
                raise RuntimeError("SIGMA window icon was replaced")
            print(json.dumps(smoke_checks(viewer, panel, args.smoke_data_dir)), flush=True)
            app.processEvents()
            canvas = viewer.window._qt_viewer.canvas.native
            if canvas.width() < 200 or canvas.height() < 120:
                raise RuntimeError(f"Image canvas is obscured: {canvas.width()}x{canvas.height()}")
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
