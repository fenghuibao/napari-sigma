"""Check real Setup/Uninstall PE icon resources, not just build configuration."""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import struct


def ico_images(path):
    raw = path.read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", raw)
    if reserved or kind != 1 or count < 5:
        raise ValueError("Expected a multi-resolution Windows icon")
    images = []
    for n in range(count):
        size, offset = struct.unpack_from("<II", raw, 6 + n * 16 + 8)
        images.append(hashlib.sha256(raw[offset:offset + size]).hexdigest())
    return sorted(images)


def pe_icon_groups(path):
    dll = ctypes.WinDLL("kernel32", use_last_error=True)
    dll.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
    dll.LoadLibraryExW.restype = wintypes.HMODULE
    dll.FindResourceW.argtypes = [wintypes.HMODULE, ctypes.c_void_p, ctypes.c_void_p]
    dll.FindResourceW.restype = wintypes.HANDLE
    dll.SizeofResource.argtypes = [wintypes.HMODULE, wintypes.HANDLE]
    dll.SizeofResource.restype = wintypes.DWORD
    dll.LoadResource.argtypes = [wintypes.HMODULE, wintypes.HANDLE]
    dll.LoadResource.restype = wintypes.HANDLE
    dll.LockResource.argtypes = [wintypes.HANDLE]
    dll.LockResource.restype = ctypes.c_void_p
    dll.FreeLibrary.argtypes = [wintypes.HMODULE]
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMODULE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ssize_t)
    dll.EnumResourceNamesW.argtypes = [wintypes.HMODULE, ctypes.c_void_p, callback_type, ctypes.c_ssize_t]
    dll.EnumResourceNamesW.restype = wintypes.BOOL
    handle = dll.LoadLibraryExW(str(path), None, 0x00000002 | 0x00000020)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    groups, errors = [], []

    def resource(name, kind):
        found = dll.FindResourceW(handle, name, kind)
        if not found:
            raise ctypes.WinError(ctypes.get_last_error())
        loaded = dll.LoadResource(handle, found)
        address = dll.LockResource(loaded)
        if not address:
            raise ctypes.WinError(ctypes.get_last_error())
        return ctypes.string_at(address, dll.SizeofResource(handle, found))

    @callback_type
    def collect(module, kind, name, param):
        try:
            group = resource(name, 14)  # RT_GROUP_ICON
            count = struct.unpack_from("<H", group, 4)[0]
            hashes = []
            for n in range(count):
                icon_id = struct.unpack_from("<H", group, 6 + n * 14 + 12)[0]
                hashes.append(hashlib.sha256(resource(icon_id, 3)).hexdigest())
            groups.append(sorted(hashes))
            return True
        except Exception as exc:
            errors.append(exc)
            return False

    try:
        ok = dll.EnumResourceNamesW(handle, 14, collect, 0)
        if errors:
            raise errors[0]
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        dll.FreeLibrary(handle)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installer", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    icon = args.prefix / "sigma-desktop/sigma.ico"
    expected = ico_images(icon)
    report = {"icon": str(icon), "image_hashes": expected, "files": {}}
    for executable in (args.installer, args.prefix / "unins000.exe"):
        if expected not in pe_icon_groups(executable.resolve()):
            raise ValueError(f"SIGMA icon is not embedded in {executable.name}")
        report["files"][executable.name] = "passed"
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Verified multi-resolution SIGMA icons in both Setup and Uninstall")


if __name__ == "__main__":
    main()
