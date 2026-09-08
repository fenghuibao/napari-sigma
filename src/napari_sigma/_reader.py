"""Lightweight reader and shared TCZYX-to-layer conversion."""
from __future__ import annotations

import os

import numpy as np

from ._image_io import SUPPORTED_IMAGE_SUFFIXES, load_image_tc_zyx


def tczyx_to_layer_data(data, meta: dict, name: str):
    array = np.asarray(data)
    if array.ndim != 5:
        raise ValueError(f"Expected TCZYX data, got {array.shape}.")
    t, channels, z, _y, _x = array.shape
    sc = tuple(meta.get("scale_per_axis", (1, 1, 1, 1, 1)))
    channel_names = meta.get("channel_names") or []
    kind = "labels" if meta.get("layer_type") == "labels" else "image"
    layers = []
    for channel in range(channels):
        volume = array[:, channel]
        if t == 1 and z == 1:
            volume, dims, scale = volume[0, 0], "YX", sc[-2:]
        elif t == 1:
            volume, dims, scale = volume[0], "ZYX", sc[-3:]
        elif z == 1:
            volume, dims, scale = volume[:, 0], "TYX", (sc[0], *sc[-2:])
        else:
            dims, scale = "TZYX", (sc[0], *sc[-3:])
        metadata = dict(meta)
        metadata.update(dims=dims, dims_out=dims, channel_index=channel)
        channel_name = channel_names[channel] if channel < len(channel_names) else f"Channel {channel + 1}"
        metadata["channel_name"] = channel_name
        kwargs = {"name": name if channels == 1 else f"{name} [{channel_name}]",
                  "metadata": metadata, "scale": scale}
        if kind == "image":
            kwargs.update(rgb=False, blending="additive")
        elif volume.dtype.kind not in "uib":
            raise ValueError("Label TIFF data must have an integer dtype.")
        if kind == "labels" and volume.dtype.kind == "b":
            volume = volume.astype(np.uint8)
        layers.append((volume, kwargs, kind))
    return layers


def napari_get_reader(paths):
    if isinstance(paths, (list, tuple)):
        if len(paths) != 1:
            return None
        path = paths[0]
    else:
        path = paths
    if not os.fspath(path).lower().endswith(SUPPORTED_IMAGE_SUFFIXES):
        return None

    def read(selected_paths):
        selected = selected_paths[0] if isinstance(selected_paths, (list, tuple)) else selected_paths
        selected = os.fspath(selected)
        data, meta = load_image_tc_zyx(selected)
        return tczyx_to_layer_data(data, meta, os.path.basename(selected))

    return read
