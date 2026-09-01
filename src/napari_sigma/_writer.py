from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from PIL import Image

_SUPPORTED_WRITE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".gif", ".mp4"}
_SIGMA_TIFF_METADATA_KEYS = (
    "sigma_layer_role",
    "sigma_saved_structure",
    "is_frangi",
    "frangi_result_name",
    "frangi_response_mode",
    "frangi_rescale_enabled",
    "frangi_rescale_low_percent",
    "frangi_vessel_rescale_low_percent",
    "frangi_sheet_rescale_low_percent",
    "frangi_sigmas",
    "frangi_vessel_sigmas",
    "frangi_sheet_sigmas",
    "frangi_combined_method",
    "frangi_combined_component",
    "is_proximity_roi_mask",
    "proximity_source_layer",
    "proximity_roi_count",
)


def _unit_from_meta(meta: dict) -> str:
    layer_meta = meta.get("metadata", {}) or {}
    return layer_meta.get("PhysicalSizeXUnit") or layer_meta.get("unit") or layer_meta.get("Units") or "um"


def _layer_type_from_meta(meta: dict) -> str:
    return str(meta.get("layer_type") or (meta.get("metadata", {}) or {}).get("layer_type") or "image")


def _fps_from_meta(meta: dict) -> float:
    layer_meta = meta.get("metadata", {}) or {}
    value = layer_meta.get("fps", meta.get("fps", 1.0))
    try:
        return max(float(value), 0.1)
    except (TypeError, ValueError):
        return 1.0


def _scale_from_meta(meta: dict, ndim: int) -> tuple[float, ...]:
    scale = tuple(float(v) for v in (meta.get("scale") or (1,) * ndim))
    if len(scale) >= ndim:
        return scale[-ndim:]
    return (1.0,) * (ndim - len(scale)) + scale


def _axes_for_ndim(ndim: int) -> str:
    return {
        2: "YX",
        3: "ZYX",
        4: "TZYX",
        5: "TCZYX",
    }.get(ndim, "".join("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[-ndim:]))


def _axes_from_meta(meta: dict, ndim: int) -> str:
    layer_meta = meta.get("metadata", {}) or {}
    candidates = [
        layer_meta.get("dims_out"),
        layer_meta.get("dims"),
        layer_meta.get("inferred_dims"),
        meta.get("dims_out"),
        meta.get("dims"),
        meta.get("inferred_dims"),
    ]
    for candidate in candidates:
        axes = str(candidate or "").upper()
        if axes and len(axes) == ndim:
            return axes
    return _axes_for_ndim(ndim)


def _coerce_tiff_dtype(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return TIFF-writable data and whether ImageJ mode can be used."""
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8, copy=False), True
    if arr.dtype.kind == "f":
        return arr.astype(np.float32, copy=False), True
    if arr.dtype.kind in {"i", "u"}:
        arr_min = int(np.min(arr))
        arr_max = int(np.max(arr))
        if arr_min >= 0 and arr_max <= np.iinfo(np.uint8).max:
            return arr.astype(np.uint8, copy=False), True
        if arr_min >= 0 and arr_max <= np.iinfo(np.uint16).max:
            return arr.astype(np.uint16, copy=False), True
        if arr_min >= np.iinfo(np.int16).min and arr_max <= np.iinfo(np.int16).max:
            return arr.astype(np.int16, copy=False), False
        if arr_min >= 0 and arr_max <= np.iinfo(np.uint32).max:
            return arr.astype(np.uint32, copy=False), False
        if arr_min >= np.iinfo(np.int32).min and arr_max <= np.iinfo(np.int32).max:
            return arr.astype(np.int32, copy=False), False
    return arr.astype(np.float32, copy=False), True


def _write_tiff(path: str, data: np.ndarray, meta: dict) -> None:
    arr, imagej_ok = _coerce_tiff_dtype(np.asarray(data))
    scale = _scale_from_meta(meta, arr.ndim)
    unit = _unit_from_meta(meta)
    axes = _axes_from_meta(meta, arr.ndim)
    write_kwargs: dict[str, Any] = {}
    is_rgb = bool(arr.ndim >= 3 and arr.shape[-1] in (3, 4) and axes.endswith("C"))

    if "Y" in axes and "X" in axes:
        y_idx = axes.index("Y")
        x_idx = axes.index("X")
        vy = float(scale[y_idx])
        vx = float(scale[x_idx])
        write_kwargs["resolution"] = (1.0 / max(vx, 1e-12), 1.0 / max(vy, 1e-12))
    if is_rgb:
        write_kwargs["photometric"] = "rgb"
        imagej_ok = False

    metadata = {"axes": axes, "unit": unit, "layer_type": _layer_type_from_meta(meta)}
    if "Z" in axes:
        metadata["spacing"] = float(scale[axes.index("Z")])
    if "T" in axes:
        layer_meta = meta.get("metadata", {}) or {}
        interval = layer_meta.get("finterval", layer_meta.get("time_interval"))
        try:
            interval = float(interval)
        except (TypeError, ValueError):
            interval = float(scale[axes.index("T")])
        if interval > 0:
            metadata["finterval"] = interval
            metadata["fps"] = 1.0 / interval
    if is_rgb:
        metadata["axes"] = axes[:-1] + "S"

    layer_meta = meta.get("metadata", {}) or {}
    sigma_metadata = {}
    for key in _SIGMA_TIFF_METADATA_KEYS:
        if key not in layer_meta:
            continue
        value = layer_meta[key]
        if isinstance(value, np.generic):
            value = value.item()
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        sigma_metadata[key] = value
    if sigma_metadata:
        metadata["sigma_metadata"] = json.dumps(sigma_metadata, separators=(",", ":"))

    tifffile.imwrite(
        path,
        arr,
        imagej=imagej_ok,
        metadata=metadata,
        **write_kwargs,
    )


def _write_gif(path: str, data: np.ndarray, meta: dict) -> None:
    arr = np.asarray(data)
    if arr.ndim != 4 or arr.shape[-1] not in (3, 4):
        raise ValueError("GIF export expects TYXC RGB(A) data.")
    fps = _fps_from_meta(meta)
    frames = [Image.fromarray(frame.astype(np.uint8, copy=False)) for frame in arr]
    duration_ms = max(int(round(1000.0 / fps)), 1)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
        disposal=2,
    )


def _write_mp4(path: str, data: np.ndarray, meta: dict) -> None:
    arr = np.asarray(data)
    if arr.ndim != 4 or arr.shape[-1] not in (3, 4):
        raise ValueError("MP4 export expects TYXC RGB(A) data.")
    frames = arr[..., :3].astype(np.uint8, copy=False)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        _fps_from_meta(meta),
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError("Could not open MP4 writer.")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _write_image(path: str, data: Any, meta: dict) -> None:
    arr = np.asarray(data)
    suffix = Path(path).suffix.lower()
    if suffix not in _SUPPORTED_WRITE_SUFFIXES:
        raise ValueError(f"Unsupported export format: {suffix}")
    if suffix in {".tif", ".tiff"}:
        _write_tiff(path, arr, meta)
        return
    if suffix == ".gif":
        _write_gif(path, arr, meta)
        return
    if suffix == ".mp4":
        _write_mp4(path, arr, meta)
        return
    Image.fromarray(arr).save(path)


def write_single_image(path: str, data: Any, meta: dict) -> list[str]:
    _write_image(path, data, meta)
    return [path]
