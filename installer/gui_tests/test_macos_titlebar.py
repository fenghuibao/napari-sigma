"""Measure the real AppKit title position, not a Qt content-area label."""
import importlib.util
import math
import os
from pathlib import Path
import sys
import unittest


@unittest.skipUnless(sys.platform == "darwin", "Native macOS title bar")
class MacTitlebarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from napari_sigma._launcher import _configure_pyqt6
        _configure_pyqt6()
        from napari._qt.qt_event_loop import get_qapp
        cls.app = get_qapp()
        resources = Path(os.environ["SIGMA_DESKTOP_TEST_RESOURCES"])
        spec = importlib.util.spec_from_file_location("titlebar_test", resources / "mac_window.py")
        cls.native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.native)

    def test_title_remains_centered_at_different_window_widths(self):
        from qtpy.QtWidgets import QMainWindow
        window = QMainWindow()
        self.addCleanup(window.close)
        title = "SIGMA (Structurely-aware Intensity-ordered GMM-MRF Algorithm)"
        window.setWindowTitle(title)
        window.show()
        self.native.center_window_title(window)
        for width in (1000, 1400, 1900):
            with self.subTest(width=width):
                window.resize(width, 480)
                self.app.processEvents()
                offset = self.native.title_center_offset(window)
                self.assertTrue(math.isfinite(offset))
                self.assertLessEqual(abs(offset), 2)
                self.assertEqual(window.windowTitle(), title)

    def test_invalid_view_is_rejected_without_native_dereference(self):
        self.assertEqual(self.native.library().SIGMACenterWindowTitle(None), 0)

    def test_cold_title_is_centered_before_next_native_event_loop_turn(self):
        from qtpy.QtWidgets import QMainWindow
        for maximized in (False, True):
            with self.subTest(maximized=maximized):
                window = QMainWindow()
                self.addCleanup(window.close)
                window.setWindowTitle("SIGMA (Structurely-aware Intensity-ordered GMM-MRF Algorithm)")
                window.resize(1400, 480)
                window.showMaximized() if maximized else window.show()
                self.native.center_window_title(window)
                # Deliberately do not call processEvents or sleep: initial
                # native constraints must already be reflected in geometry.
                offset = self.native.title_center_offset(window)
                self.assertTrue(math.isfinite(offset))
                self.assertLessEqual(abs(offset), 2)
