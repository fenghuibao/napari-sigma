"""Exercise the installed desktop adapter and real Qt event loop."""
import importlib.util
import os
from pathlib import Path
import sys
import threading
import time
import unittest

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
    def make_panel(self, preparation):
        viewer = napari.Viewer(show=False)
        panel = desktop.DesktopSIGMAWidget(viewer, plot_preparation=preparation)
        viewer.window.add_dock_widget(panel, name="SIGMA")
        self.addCleanup(viewer.close)
        self.addCleanup(panel.close)
        self.addCleanup(panel.dispose)
        return viewer, panel

    def process_until(self, condition, timeout=20):
        end = time.monotonic() + timeout
        while not condition() and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(.01)
        self.assertTrue(condition(), "Condition was not reached while processing Qt events")

    def test_pending_import_keeps_ui_responsive_and_creates_canvas_on_main_thread(self):
        release = threading.Event()
        thread_ids = []
        def prepare():
            thread_ids.append(threading.get_ident())
            if not release.wait(10):
                raise RuntimeError("Test did not release background preparation")
            desktop.prepare_plot_modules()
        preparation = desktop.PlotPreparation(prepare)
        self.addCleanup(release.set)
        viewer, panel = self.make_panel(preparation)
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
        self.process_until(lambda: panel._analysis_plot_canvas is not None)
        self.assertIsNone(preparation.error)
        self.assertNotEqual(thread_ids[0], threading.get_ident())
        self.assertEqual(panel._analysis_plot_canvas.thread(), QApplication.instance().thread())
        self.assertFalse(panel._desktop_plot_timer.isActive())

    def test_close_while_preparing_does_not_wait_or_touch_deleted_widgets(self):
        release = threading.Event()
        preparation = desktop.PlotPreparation(lambda: release.wait(10))
        self.addCleanup(release.set)
        viewer, panel = self.make_panel(preparation)
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
        viewer, panel = self.make_panel(preparation)
        panel._panel_tabs.setCurrentWidget(panel._analysis_tab)
        self.process_until(lambda: panel._desktop_plot_error is not None)
        self.assertIn("test font load failure", panel._analysis_plot_placeholder.text())
        panel._panel_tabs.setCurrentIndex(0)
        self.assertEqual(panel._panel_tabs.currentIndex(), 0)
        self.assertIsNone(panel._analysis_plot_canvas)

    def test_default_preparation_is_shared_between_panels(self):
        self.assertIs(desktop.shared_preparation(), desktop.shared_preparation())


if __name__ == "__main__":
    unittest.main()
