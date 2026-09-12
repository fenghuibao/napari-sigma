"""Give the disk image and its mounted volume the SIGMA emblem.

Finder draws a custom icon only when an item carries the kHasCustomIcon flag in
its ``com.apple.FinderInfo`` attribute. A volume then reads the artwork from
``.VolumeIcon.icns`` at its root, while a plain file reads an ``icns`` resource
from its resource fork. Both structures are written here directly, so the build
does not depend on the deprecated Rez/DeRez/SetFile developer tools.

A resource fork is filesystem metadata. It survives a local copy or ``ditto``
but not a zip, a web download or a GitHub artifact, so the icon on the .dmg
file itself is a convenience for locally produced images. The volume icon lives
inside the disk image and is therefore always preserved.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import struct
import subprocess

# Finder reads a file's custom icon from the 'icns' resource with this ID.
CUSTOM_ICON_RESOURCE_ID = -16455
HAS_CUSTOM_ICON = 0x0400
FINDER_INFO_SIZE = 32
RESOURCE_HEADER_SIZE = 256
VOLUME_ICON_NAME = ".VolumeIcon.icns"


def finder_info(flags: int = HAS_CUSTOM_ICON) -> bytes:
    """FinderInfo whose only set bits are ``flags``.

    FileInfo and FolderInfo both place the 16-bit Finder flags at offset 8, so
    one layout serves files, folders and volume roots. Type, creator and
    position fields stay zero.
    """
    return b"\0" * 8 + struct.pack(">H", flags) + b"\0" * (FINDER_INFO_SIZE - 10)


def resource_fork(icns: bytes) -> bytes:
    """A resource fork holding exactly one custom-icon 'icns' resource."""
    data = struct.pack(">I", len(icns)) + icns
    # Type list: one type, whose reference list follows it immediately.
    type_list = struct.pack(">H", 0) + b"icns" + struct.pack(">HH", 0, 10)
    # Reference: id, name offset (-1 for unnamed), attributes, 24-bit offset
    # into the data section, and a zero placeholder handle.
    reference = (struct.pack(">hhB", CUSTOM_ICON_RESOURCE_ID, -1, 0)
                 + (0).to_bytes(3, "big") + struct.pack(">I", 0))
    type_list_offset = 28
    name_list_offset = type_list_offset + len(type_list) + len(reference)
    resource_map = (b"\0" * 16
                    + struct.pack(">IHHHH", 0, 0, 0, type_list_offset, name_list_offset)
                    + type_list + reference)
    header = struct.pack(">IIII", RESOURCE_HEADER_SIZE, RESOURCE_HEADER_SIZE + len(data),
                         len(data), len(resource_map))
    return header.ljust(RESOURCE_HEADER_SIZE, b"\0") + data + resource_map


def mark_custom_icon(path: Path):
    """Set kHasCustomIcon on a file, folder or mounted volume root."""
    subprocess.run(["/usr/bin/xattr", "-wx", "com.apple.FinderInfo",
                    finder_info().hex(), str(path)], check=True)


def stage_volume_icon(staging: Path, icns: Path):
    """Place the artwork a mounted volume will display as its icon."""
    shutil.copy2(icns, staging / VOLUME_ICON_NAME)


def set_file_icon(path: Path, icns: Path):
    """Attach the artwork to a single file, such as the finished .dmg."""
    Path(f"{path}/..namedfork/rsrc").write_bytes(resource_fork(icns.read_bytes()))
    mark_custom_icon(path)
