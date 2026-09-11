"""Launch napari with a deterministic PyQt6 runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path


class QtSetupError(RuntimeError):
    """Raised when a usable PyQt6 runtime cannot be configured."""


# Extensions SIGMA's own reader handles, per its npe2 manifest.
SIGMA_READER_PATTERNS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")


PLUGIN_NAME = "napari-sigma"
# Readers whose claim on a pattern SIGMA will take over. napari's built-in
# reader is the one that produces the uncalibrated layers described below;
# anything else is a deliberate third-party choice and is left alone.
_REPLACEABLE_READERS = ("napari", "builtins", PLUGIN_NAME)


def prefer_sigma_reader(patterns=SIGMA_READER_PATTERNS, *, force: bool = False) -> dict:
    """Make SIGMA's reader the one that opens these file patterns.

    Returns ``{"changed": {...}, "kept": {...}, "mapping": {...}}``.

    Why this is necessary and not merely tidier: napari's built-in reader
    creates the layer with no units and no SIGMA metadata, and napari's own
    ``inserted`` callback reaches a canvas draw *synchronously inside*
    ``LayerList.insert``::

        layerlist.py insert() -> event.py __call__()
          -> qt_viewer.py _on_add_layer_change() -> _add_layer()
          -> canvas.py add_layer_visual_mapping() -> _update_scenegraph()
          -> canvas.py on_draw() -> canvas.py _update_world_units()

    That draw runs the unit-consistency check while the new layer is still on
    napari's default "pixel", whose dimensionality is ``[printing_unit]``
    rather than ``[length]``, so ``layers.extent.units`` is None and napari
    reports "Inconsistent units across layers; units will not be used for
    rendering" - once per dropped file, even though every file declares
    micrometres and SIGMA calibrates the layer milliseconds later.

    A plugin cannot get ahead of that draw. napari sorts callbacks so that its
    own always run first: ``EventEmitter.connect`` partitions on
    ``_is_core_callback``, i.e. ``callback.__module__.startswith('napari.')``,
    and forces non-core callbacks after every core one, so ``position='first'``
    only orders within the plugin region. ``napari_sigma.`` is not ``napari.``.
    The layer therefore has to *arrive* calibrated, which is exactly what
    SIGMA's reader does (``_reader.tczyx_to_layer_data`` passes ``units`` at
    construction) - so the fix is to make sure it is the reader that runs.

    A pattern already assigned to another plugin is preserved unless ``force``,
    because that assignment is a choice someone made; an unset pattern, or one
    pointing at napari's built-in reader, is claimed.
    """
    from napari.settings import get_settings

    settings = get_settings()
    current = dict(settings.plugins.extension2reader)
    changed, kept = {}, {}
    for pattern in patterns:
        existing = current.get(pattern)
        if force or existing is None or existing in _REPLACEABLE_READERS:
            if existing != PLUGIN_NAME:
                changed[pattern] = existing
            current[pattern] = PLUGIN_NAME
        else:
            kept[pattern] = existing
    if changed:
        # Reassign rather than mutate in place: the settings model emits and
        # persists on attribute assignment. Only when something differs, so
        # opening the panel does not rewrite the settings file every time.
        settings.plugins.extension2reader = current
    return {
        "changed": changed,
        "kept": kept,
        "mapping": dict(settings.plugins.extension2reader),
    }


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
