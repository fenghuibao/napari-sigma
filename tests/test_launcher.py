from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

from napari_sigma._launcher import (
    QtSetupError,
    _platform_plugin_name,
    _prepare_environment,
)


class LauncherEnvironmentTests(unittest.TestCase):
    def test_prepare_environment_selects_pyqt6_and_clears_plugin_paths(self):
        environment = {
            "QT_API": "pyside6",
            "QT_PLUGIN_PATH": "/old/plugins",
            "QT_QPA_PLATFORM_PLUGIN_PATH": "/old/platforms",
        }
        with patch.dict(os.environ, environment, clear=True):
            _prepare_environment()
            self.assertEqual(os.environ["QT_API"], "pyqt6")
            self.assertNotIn("QT_PLUGIN_PATH", os.environ)
            self.assertNotIn("QT_QPA_PLATFORM_PLUGIN_PATH", os.environ)

    def test_prepare_environment_rejects_loaded_binding(self):
        with patch.dict(sys.modules, {"PySide6": object()}):
            with self.assertRaisesRegex(QtSetupError, "already loaded"):
                _prepare_environment()

    def test_platform_plugin_name_is_defined(self):
        self.assertTrue(_platform_plugin_name())


if __name__ == "__main__":
    unittest.main()
