"""Launch napari with a deterministic PyQt6 runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path


class QtSetupError(RuntimeError):
    """Raised when a usable PyQt6 runtime cannot be configured."""


def _platform_plugin_name() -> str:
    if sys.platform == "darwin":
        return "libqcocoa.dylib"
    if sys.platform == "win32":
        return "qwindows.dll"
    return "libqxcb.so"


def _prepare_environment() -> None:
    loaded = [
        binding
        for binding in ("PyQt5", "PySide2", "PySide6")
        if binding in sys.modules
    ]
    if loaded:
        names = ", ".join(loaded)
        raise QtSetupError(
            f"another Qt binding is already loaded ({names}); start "
            "napari-sigma in a fresh process"
        )

    os.environ["QT_API"] = "pyqt6"
    # Ignore inherited plugin paths from a different Qt installation. This
    # changes only the launcher process, not the user's shell configuration.
    os.environ.pop("QT_PLUGIN_PATH", None)
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


def _configure_pyqt6() -> Path:
    _prepare_environment()

    try:
        import PyQt6
        from PyQt6.QtCore import QCoreApplication, QLibraryInfo
    except (ImportError, ModuleNotFoundError) as exc:
        raise QtSetupError(
            "PyQt6 is not installed in this Python environment. Install it "
            'with: python -m pip install --upgrade "napari-sigma[all]"'
        ) from exc

    bundled_plugins = Path(PyQt6.__file__).resolve().parent / "Qt6" / "plugins"
    plugin_name = _platform_plugin_name()

    if (bundled_plugins / "platforms" / plugin_name).is_file():
        plugin_root = bundled_plugins
    else:
        plugin_root = Path(
            QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
        )
        if not (plugin_root / "platforms" / plugin_name).is_file():
            raise QtSetupError(
                f"the {plugin_name} Qt platform plugin was not found. "
                "Create a clean environment and reinstall napari-sigma[all]"
            )

    # Set the in-process search path before QApplication is constructed. This
    # avoids stale conda qt.conf files redirecting PyQt6 to Qt5 plugins.
    QCoreApplication.setLibraryPaths([str(plugin_root)])
    return plugin_root


def main() -> int:
    """Run napari after selecting and validating the PyQt6 backend."""
    try:
        _configure_pyqt6()
    except QtSetupError as exc:
        print(f"napari-sigma: {exc}", file=sys.stderr)
        return 1

    from napari.__main__ import main as napari_main

    napari_main()
    return 0


__all__ = ["main"]
