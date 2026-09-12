"""Exercise the installed desktop adapter and real Qt event loop."""
import importlib.util
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

from napari_sigma._launcher import _configure_pyqt6
_configure_pyqt6()
import napari
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication

resources = Path(os.environ.get("SIGMA_DESKTOP_TEST_RESOURCES", Path(sys.prefix) / "sigma-desktop"))
spec = importlib.util.spec_from_file_location("sigma_desktop_test", resources / "desktop_widget.py")
desktop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(desktop)


class PlotPreparationTests(unittest.TestCase):
    FULL_NAME = "SIGMA (Structurely-aware Intensity-ordered GMM-MRF Algorithm)"

    def test_full_name_titles_the_panel_without_changing_the_app_name(self):
        preparation = desktop.PlotPreparation(lambda: None)
        viewer, panel, dock = self.make_panel(preparation)
        self.assertEqual(desktop.FULL_NAME, self.FULL_NAME)
        self.assertEqual(dock.windowTitle(), self.FULL_NAME)
        self.assertEqual(panel.windowTitle(), self.FULL_NAME)
        self.assertNotIn("0.0.5", dock.windowTitle())
        # The name belongs to the title bars, not to a row above the panel.
        # The scroll area holding the tabs is first again, as it is upstream.
        self.assertFalse(hasattr(panel, "_desktop_full_name"))
        self.assertIs(panel.layout().itemAt(0).widget(), panel._scroll)
        self.assertEqual(panel.layout().count(), 1)
        self.assertEqual(QApplication.instance().applicationDisplayName(), "SIGMA")

    def test_window_title_carries_the_full_name(self):
        preparation = desktop.PlotPreparation(lambda: None)
        viewer, panel, dock = self.make_panel(preparation)
        viewer.title = desktop.FULL_NAME
        self.assertIn(self.FULL_NAME, viewer.window._qt_window.windowTitle())

    def make_panel(self, preparation):
        viewer = napari.Viewer(show=False)
        # Viewer creates the application's first Qt instance in a fresh test
        # process. Do not assume another test has already constructed one.
        QApplication.instance().setApplicationDisplayName("SIGMA")
        panel = desktop.DesktopSIGMAWidget(viewer, plot_preparation=preparation)
        dock = viewer.window.add_dock_widget(panel, name=desktop.FULL_NAME)
        self.addCleanup(viewer.close)
        self.addCleanup(panel.close)
        self.addCleanup(panel.dispose)
        return viewer, panel, dock

    def process_until(self, condition, timeout=20, diagnostics=None):
        end = time.monotonic() + timeout
        while not condition() and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(.01)
        if not condition():
            details = diagnostics() if diagnostics is not None else ""
            self.fail(f"Condition not reached after {timeout}s while processing Qt events. {details}")

    def test_pending_import_keeps_ui_responsive_and_creates_canvas_on_main_thread(self):
        release = threading.Event()
        self.addCleanup(release.set)
        thread_ids = []
        def prepare():
            thread_ids.append(threading.get_ident())
            # Cleanup always releases this daemon worker, including on failure.
            # Slow panel construction must not accidentally open the test gate.
            release.wait()
            desktop.prepare_plot_modules()
        preparation = desktop.PlotPreparation(prepare)
        viewer, panel, _ = self.make_panel(preparation)
        beats = []
        timer = QTimer(panel)
        timer.setInterval(10)
        timer.timeout.connect(lambda: beats.append(1))
        timer.start()
        panel._panel_tabs.setCurrentWidget(panel._analysis_tab)
        self.assertIsNone(panel._analysis_plot_axes)
        self.assertIn("background", panel._analysis_plot_placeholder.text())
        self.process_until(lambda: len(beats) >= 3)
        self.assertFalse(preparation.done.is_set())
        panel._panel_tabs.setCurrentIndex(0)
        panel._panel_tabs.setCurrentWidget(panel._analysis_tab)
        self.assertIsNone(panel._analysis_plot_axes)
        release.set()
        # Cold system font enumeration can exceed 20s on the Intel CI runner.
        # This is a completion budget, not permission to block the GUI: the
        # event-loop assertions above still run while preparation is gated.
        self.process_until(
            lambda: panel._analysis_plot_canvas is not None or preparation.error is not None,
            timeout=180,
            diagnostics=lambda: (
                f"worker_done={preparation.done.is_set()}, error={preparation.error!r}, "
                f"preparation_seconds={preparation.seconds}, Qt_ticks={len(beats)}, "
                f"placeholder={panel._analysis_plot_placeholder.text()!r}"),
        )
        self.assertIsNone(preparation.error)
        self.assertIsNotNone(panel._analysis_plot_canvas)
        self.assertNotEqual(thread_ids[0], threading.get_ident())
        self.assertEqual(panel._analysis_plot_canvas.thread(), QApplication.instance().thread())
        self.assertFalse(panel._desktop_plot_timer.isActive())
        print(f"Background plot preparation: {preparation.seconds:.3f}s; Qt ticks: {len(beats)}", flush=True)

    def test_close_while_preparing_does_not_wait_or_touch_deleted_widgets(self):
        release = threading.Event()
        self.addCleanup(release.set)
        preparation = desktop.PlotPreparation(release.wait)
        viewer, panel, _ = self.make_panel(preparation)
        panel.dispose()
        self.assertFalse(panel._desktop_plot_timer.isActive())
        self.assertFalse(preparation.done.is_set())
        release.set()
        self.process_until(preparation.done.is_set)
        self.assertIsNone(panel._analysis_plot_canvas)

    def test_failed_import_is_visible_without_disabling_other_tabs(self):
        def fail():
            raise ImportError("test font load failure")
        preparation = desktop.PlotPreparation(fail)
        viewer, panel, _ = self.make_panel(preparation)
        panel._panel_tabs.setCurrentWidget(panel._analysis_tab)
        self.process_until(lambda: panel._desktop_plot_error is not None)
        self.assertIn("test font load failure", panel._analysis_plot_placeholder.text())
        panel._panel_tabs.setCurrentIndex(0)
        self.assertEqual(panel._panel_tabs.currentIndex(), 0)
        self.assertIsNone(panel._analysis_plot_canvas)

    def test_default_preparation_is_shared_between_panels(self):
        # Test singleton ownership without launching a second real importer
        # that interferes with the separate cold-font integration test.
        with patch.object(desktop, "_shared_preparation", None), \
                patch.object(desktop, "PlotPreparation") as constructor:
            self.assertIs(desktop.shared_preparation(), constructor.return_value)
            self.assertIs(desktop.shared_preparation(), constructor.return_value)
            constructor.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
