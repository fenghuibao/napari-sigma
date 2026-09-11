"""Responsive desktop plotting preparation for the published SIGMA widget.

Only non-GUI module/font initialization runs in a worker. Every Figure, Artist
and Qt canvas is still created and used exclusively on the GUI thread.
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import threading
from time import perf_counter

from qtpy.QtCore import Qt, QThread, QTimer
from qtpy.QtWidgets import QLabel, QSizePolicy
from napari_sigma._widget import SIGMAWidget

FULL_NAME = "SIGMA (Structurely-aware Intensity-ordered GMM-MRF Algorithm)"


def prepare_plot_modules():
    # No pyplot, Qt backend selection, figures, or GUI objects in this thread.
    # Restore the bundled-only index before anything can initialize font_manager.
    # -I excludes the script directory: load the helper by its trusted file path.
    resources = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("sigma_fonts", resources / "font_cache.py")
    fonts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fonts)
    fonts.prepare_font_cache(resources)
    importlib.import_module("matplotlib.figure")
    importlib.import_module("matplotlib.backends.backend_agg")


class PlotPreparation:
    def __init__(self, prepare=prepare_plot_modules):
        self.done = threading.Event()
        self.error = None
        self.seconds = None
        # The worker owns no widget or Qt object. Closing a window never waits
        # for system font enumeration and cannot leave a running QThread behind.
        self.thread = threading.Thread(target=self._run, args=(prepare,),
                                       name="SIGMA-plot-preparation", daemon=True)
        self.thread.start()

    def _run(self, prepare):
        start = perf_counter()
        try:
            prepare()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.seconds = perf_counter() - start
            self.done.set()


_shared_preparation = None
_preparation_lock = threading.Lock()


def shared_preparation():
    global _shared_preparation
    with _preparation_lock:
        if _shared_preparation is None:
            _shared_preparation = PlotPreparation()
        return _shared_preparation


class DesktopSIGMAWidget(SIGMAWidget):
    def __init__(self, viewer, *, plot_preparation=None):
        self._desktop_plot_ready = False
        self._desktop_plot_error = None
        self._desktop_plot_timer = None
        super().__init__(viewer)
        self._desktop_full_name = QLabel(FULL_NAME, self)
        self._desktop_full_name.setObjectName("sigmaFullName")
        self._desktop_full_name.setWordWrap(True)
        self._desktop_full_name.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._desktop_full_name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._desktop_full_name.setContentsMargins(4, 4, 4, 8)
        self.layout().insertWidget(0, self._desktop_full_name)
        self._desktop_plot_preparation = plot_preparation or shared_preparation()
        self._analysis_plot_placeholder.setText(
            "Preparing charts with bundled fonts in the background… You can keep using SIGMA.")
        self._desktop_plot_timer = QTimer(self)
        self._desktop_plot_timer.setInterval(50)
        self._desktop_plot_timer.timeout.connect(self._poll_plot_preparation)
        self._desktop_plot_timer.start()

    def _ensure_analysis_plot(self):
        # Crucially, do not enter a Matplotlib import on the GUI thread while
        # the worker holds an import lock or is scanning the system fonts.
        if not self._desktop_plot_ready:
            return False
        if QThread.currentThread() != self.thread():
            raise RuntimeError("SIGMA charts must be created on the GUI thread")
        return super()._ensure_analysis_plot()

    def _poll_plot_preparation(self):
        if self._disposed:
            self._desktop_plot_timer.stop()
            return
        preparation = self._desktop_plot_preparation
        if not preparation.done.is_set():
            return
        self._desktop_plot_timer.stop()
        self._desktop_plot_error = preparation.error
        if preparation.error is not None:
            self._analysis_plot_placeholder.setText(
                f"Charts unavailable: {preparation.error}. Restart SIGMA to retry.")
            return
        self._desktop_plot_ready = True
        if self._panel_tabs.currentWidget() is self._analysis_tab:
            # Includes any analysis data produced while imports were pending.
            self._on_panel_tab_changed()

    def showEvent(self, event):
        super().showEvent(event)
        # SIGMA permits hiding/closing and reattaching the same dock widget.
        if self._desktop_plot_timer is not None and not self._desktop_plot_ready:
            self._desktop_plot_timer.start()

    def dispose(self):
        if self._desktop_plot_timer is not None:
            self._desktop_plot_timer.stop()
        super().dispose()
