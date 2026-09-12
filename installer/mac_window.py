"""Small AppKit bridge for the macOS desktop window; no extra dependencies."""
import ctypes
from pathlib import Path
import sys

_library = None


def library():
    global _library
    if _library is None:
        _library = ctypes.CDLL(str(Path(__file__).with_name("mac_titlebar.dylib")))
        _library.SIGMACenterWindowTitle.argtypes = [ctypes.c_void_p]
        _library.SIGMACenterWindowTitle.restype = ctypes.c_int
        _library.SIGMAWindowTitleOffset.argtypes = [ctypes.c_void_p]
        _library.SIGMAWindowTitleOffset.restype = ctypes.c_double
    return _library


def center_window_title(window):
    if sys.platform == "darwin" and not library().SIGMACenterWindowTitle(int(window.winId())):
        raise RuntimeError("Could not configure the native SIGMA title bar")


def title_center_offset(window):
    return library().SIGMAWindowTitleOffset(int(window.winId()))
