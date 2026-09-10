from __future__ import annotations

import json
from contextlib import suppress
from typing import Any
from xml.etree import ElementTree

import numpy as np
import tifffile
from PIL import Image

from ._nis_tiff import read_nis_elements_tiff_metadata

SUPPORTED_IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
_UNIT_TO_UM = {
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "nm": 1e-3,
    "mm": 1e3,
    "cm": 1e4,
    "inch": 25400.0,
    "in": 25400.0,
}
_TIME_UNIT_TO_SECONDS = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "µs": 1e-6,
    "ns": 1e-9,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
}


def rescale_0_255_tczyx(arr: np.ndarray) -> np.ndarray:
    """Linearly rescale to uint8 [0..255] per (T,C) block across Z,Y,X."""
    a = np.asarray(arr)
    if a.ndim == 5:
        mins = a.min(axis=(-3, -2, -1), keepdims=True).astype(np.float32)
        maxs = a.max(axis=(-3, -2, -1), keepdims=True).astype(np.float32)
        rng = maxs - mins
        rng[rng == 0] = 1.0
        out = (a.astype(np.float32) - mins) / rng * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)
    if a.ndim == 4:
        a5 = a[None, ...]
        return rescale_0_255_tczyx(a5)[0]
    if a.ndim == 3:
        mn, mx = float(a.min()), float(a.max())
        rng = (mx - mn) if (mx > mn) else 1.0
        out = (a.astype(np.float32) - mn) / rng * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)
    return np.clip(a, 0, 255).astype(np.uint8)


def _normalize_unit_name(unit: str | None) -> str:
    if not unit:
        return "um"
    raw = str(unit).strip()
    if "\\u" in raw:
        with suppress(UnicodeDecodeError):
            raw = raw.encode("utf-8").decode("unicode_escape")
    norm = raw.lower()
    return "um" if norm in {"µm", "micron", "microns"} else norm


def _rational_to_float(value) -> float | None:
    try:
        if isinstance(value, tuple) and len(value) == 2:
            num, den = value
            return float(num) / float(den) if float(den) != 0 else None
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _pixel_size_from_resolution(resolution, unit: str | None) -> float | None:
    res = _rational_to_float(resolution)
    if res is None or res <= 0:
        return None
    scale_um = _UNIT_TO_UM.get(_normalize_unit_name(unit))
    if scale_um is None:
        return None
    return scale_um / res


def _tiff_resolution_unit_name(page, imagej_meta: dict[str, Any] | None) -> str:
    if imagej_meta and imagej_meta.get("unit"):
        return _normalize_unit_name(imagej_meta.get("unit"))
    res_unit_tag = page.tags.get("ResolutionUnit")
    res_unit = getattr(res_unit_tag, "value", None)
    if res_unit == 2:
        return "inch"
    if res_unit == 3:
        return "cm"
    return "um"


def _read_ome_pixels_metadata(
    ome_metadata: str | None,
) -> tuple[dict[str, float], float | None]:
    """Return OME physical sizes in micrometers and time interval in seconds."""
    if not ome_metadata:
        return {}, None

    try:
        root = ElementTree.fromstring(ome_metadata)
    except ElementTree.ParseError:
        return {}, None

    pixels = root.find(".//{*}Pixels")
    if pixels is None:
        return {}, None

    physical_sizes: dict[str, float] = {}
    for axis in "XYZ":
        value = pixels.get(f"PhysicalSize{axis}")
        unit = _normalize_unit_name(pixels.get(f"PhysicalSize{axis}Unit") or "um")
        scale_um = _UNIT_TO_UM.get(unit)
        try:
            size_um = float(value) * scale_um if value is not None and scale_um is not None else None
        except (TypeError, ValueError):
            size_um = None
        if size_um is not None and size_um > 0:
            physical_sizes[axis] = size_um

    time_interval = None
    time_value = pixels.get("TimeIncrement")
    time_unit = _normalize_unit_name(pixels.get("TimeIncrementUnit") or "s")
    time_scale = _TIME_UNIT_TO_SECONDS.get(time_unit)
    try:
        if time_value is not None and time_scale is not None:
            time_interval = float(time_value) * time_scale
    except (TypeError, ValueError):
        time_interval = None
    if time_interval is not None and time_interval <= 0:
        time_interval = None

    return physical_sizes, time_interval


def _read_tiff_scale_metadata(path: str) -> tuple[tuple[float, float, float], str, dict[str, Any]]:
    zyx_scale = (1.0, 1.0, 1.0)
    unit = "um"
    extras: dict[str, Any] = {}
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        # tifffile stores non-ImageJ metadata in shaped JSON (e.g. uint32 labels).
        shaped = tif.shaped_metadata or ()
        imagej_meta = dict(shaped[0] or {}) if shaped else {}
        imagej_meta.update(tif.imagej_metadata or {})
        ome_sizes, ome_time_interval = _read_ome_pixels_metadata(tif.ome_metadata)
        extras["layer_type"] = imagej_meta.get("layer_type")
        for metadata_key in ("sigma_metadata", "bistate_metadata"):
            serialized_metadata = imagej_meta.get(metadata_key)
            if not serialized_metadata:
                continue
            with suppress(json.JSONDecodeError, TypeError, ValueError):
                decoded = json.loads(serialized_metadata)
                if isinstance(decoded, dict):
                    extras.update({key: value for key, value in decoded.items()
                                   if key not in {"dims", "dims_out", "scale_per_axis", "storage_dims", "source", "axes", "layer_type"}})
        if "sigma_layer_role" not in extras and "bistate_layer_role" in extras:
            extras["sigma_layer_role"] = extras.pop("bistate_layer_role")
        if "sigma_saved_structure" not in extras and "bistate_saved_structure" in extras:
            extras["sigma_saved_structure"] = extras.pop("bistate_saved_structure")
        if extras.get("sigma_layer_role") == "structural_response":
            extras["is_frangi"] = True

        nis_xy_size, nis_extras = read_nis_elements_tiff_metadata(page)
        extras.update(nis_extras)

        source_unit = _tiff_resolution_unit_name(page, imagej_meta)
        source_scale_um = _UNIT_TO_UM.get(source_unit)
        unit = "um" if source_scale_um is not None else source_unit

        xres_tag = page.tags.get("XResolution")
        yres_tag = page.tags.get("YResolution")
        x_size = _pixel_size_from_resolution(getattr(xres_tag, "value", None), source_unit)
        y_size = _pixel_size_from_resolution(getattr(yres_tag, "value", None), source_unit)
        if "X" in ome_sizes:
            x_size = ome_sizes["X"]
            unit = "um"
        if "Y" in ome_sizes:
            y_size = ome_sizes["Y"]
            unit = "um"
        if nis_xy_size is not None:
            # NIS stores its calibrated pixel size in micrometers. Prefer it
            # over TIFF's often auto-generated X/YResolution=1 with no unit.
            x_size = nis_xy_size
            y_size = nis_xy_size
            unit = "um"

        try:
            z_size = float(imagej_meta.get("spacing"))
        except (TypeError, ValueError):
            z_size = None

        if z_size is not None and source_scale_um is not None:
            z_size *= source_scale_um
        if "Z" in ome_sizes:
            z_size = ome_sizes["Z"]
            unit = "um"

        try:
            time_interval = float(imagej_meta.get("finterval"))
        except (TypeError, ValueError):
            time_interval = None
        try:
            fps = float(imagej_meta.get("fps"))
        except (TypeError, ValueError):
            fps = None
        if (time_interval is None or time_interval <= 0) and fps is not None and fps > 0:
            time_interval = 1.0 / fps
        if ome_time_interval is not None:
            time_interval = ome_time_interval
        if time_interval is not None and time_interval > 0:
            extras["time_interval"] = float(time_interval)
            extras["finterval"] = float(time_interval)
            extras["fps"] = float(1.0 / time_interval)

        zyx_scale = (
            z_size if z_size and z_size > 0 else 1.0,
            y_size if y_size and y_size > 0 else 1.0,
            x_size if x_size and x_size > 0 else 1.0,
        )
    return zyx_scale, unit, extras


def _rgb_channels_are_identical(data: np.ndarray) -> bool:
    arr = np.asarray(data)
    if arr.ndim < 3 or arr.shape[-1] < 3:
        return False
    base = arr[..., 0]
    return bool(
        np.array_equal(base, arr[..., 1]) and np.array_equal(base, arr[..., 2])
    )


def _is_probable_time_series(imagej_meta: dict[str, Any] | None) -> bool:
    meta = imagej_meta or {}
    if meta.get("frames") not in (None, 0, 1):
        return True
    if meta.get("fps") not in (None, 0):
        return True
    labels = meta.get("Labels")
    if not labels:
        return False
    first = str(labels[0]).strip().lower()
    if first.startswith("z:"):
        return False
    if first.startswith("t:"):
        return True
    return False


def _interpret_tiff_axes(
    axes: str, ndim: int, imagej_meta: dict[str, Any] | None = None,
) -> tuple[str, dict[str, str]]:
    """Resolve an unlabelled page axis; keep assumptions separate from metadata."""
    axes = axes.upper()
    original = axes
    if len(axes) != ndim or len(set(axes)) != ndim or not {"Y", "X"} <= set(axes):
        raise ValueError(f"Unsupported TIFF axes layout: {axes} for {ndim} dimensions")
    assumptions = {}
    page_axes = set(axes) & {"I", "Q"}
    if len(page_axes) == 1 and "Z" not in axes:
        page_axis = page_axes.pop()
        # Without acquisition metadata a page index cannot establish time vs Z.
        # Default to Z, and expose this assumption in the returned file metadata.
        inferred = "T" if "T" not in axes and _is_probable_time_series(imagej_meta) else "Z"
        assumptions[page_axis] = inferred
        axes = axes.replace(page_axis, inferred)
    elif axes in {"ZYX", "ZYXS", "ZYXC"} and _is_probable_time_series(imagej_meta):
        # Retain support for legacy ImageJ movies stored as Z stacks.
        assumptions["Z"] = "T"
        axes = axes.replace("Z", "T")
    if not set(axes) <= set("TCZYXS"):
        raise ValueError(f"Unsupported or ambiguous TIFF axes layout: {original}")
    return axes, assumptions


def _normalize_tiff_data_to_tczyx(
    data: np.ndarray,
    axes: str,
    imagej_meta: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[str], str]:
    """Transpose named axes, never infer channel/Z order from dimension sizes."""
    axes, _ = _interpret_tiff_axes(axes, data.ndim, imagej_meta)
    # Legacy exports used a trailing C for RGB samples; TIFF itself uses S.
    if axes.endswith("C") and "S" not in axes and data.shape[-1] in (3, 4):
        axes = axes[:-1] + "S"

    sample_names = None
    if "S" in axes:
        sample_index = axes.index("S")
        samples = np.moveaxis(data, sample_index, -1)
        if samples.shape[-1] in (3, 4):
            if "C" not in axes and _rgb_channels_are_identical(samples):
                data = samples[..., 0]
                axes = axes.replace("S", "")
            else:
                # Preserve SIGMA's existing RGB convention: discard alpha.
                data = np.moveaxis(samples[..., :3], -1, sample_index)
                sample_names = ["Red", "Green", "Blue"]

    if "S" in axes and "C" not in axes:
        axes = axes.replace("S", "C")
    target_axes = "TCSZYX" if "S" in axes else "TCZYX"
    ordered_axes = "".join(axis for axis in target_axes if axis in axes)
    normalized = data.transpose(tuple(axes.index(axis) for axis in ordered_axes))
    for index, axis in enumerate(target_axes):
        if axis not in axes:
            normalized = np.expand_dims(normalized, index)
    if "S" in axes:
        # Multiple logical channels can each contain RGB samples. Flatten only
        # those two channel axes, keeping time and spatial axes untouched.
        t, c, s, z, y, x = normalized.shape
        names = [f"Channel {i + 1} {name}" for i in range(c)
                 for name in (sample_names or [f"Sample {j + 1}" for j in range(s)])]
        normalized = normalized.reshape(t, c * s, z, y, x)
    else:
        names = sample_names or [f"Channel {i + 1}" for i in range(normalized.shape[1])]
    dims = "".join(axis for axis in "TCZYX" if axis in axes)
    return normalized, names, dims


def load_image_tc_zyx(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Load common image formats and normalize them to TCZYX."""
    axes = None
    if path.lower().endswith((".tif", ".tiff")):
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            try:
                data = series.asarray(out="memmap")
            except (TypeError, ValueError, OSError, RuntimeError):
                data = series.asarray()
            axes = str(series.axes).upper()
            imagej_meta = tif.imagej_metadata or {}
            _, axis_assumptions = _interpret_tiff_axes(axes, data.ndim, imagej_meta)
            data, ch_names, inferred_dims = _normalize_tiff_data_to_tczyx(data, axes, imagej_meta)
        zyx_scale, unit, extras = _read_tiff_scale_metadata(path)
        if axis_assumptions:
            extras["axis_assumptions"] = axis_assumptions
    else:
        inferred_dims = None
        data = np.asarray(Image.open(path))
        unit = "um"
        zyx_scale = (1.0, 1.0, 1.0)
        extras = {}

        if data.ndim == 2:
            data = data[np.newaxis, np.newaxis, np.newaxis, :, :]
            ch_names = ["Channel 1"]
            inferred_dims = "YX"
        elif data.ndim == 3:
            if data.shape[-1] in (3, 4):
                rgb = data[..., :3]
                data = np.moveaxis(rgb, -1, 0)[np.newaxis, :, np.newaxis, :, :]
                ch_names = ["Red", "Green", "Blue"]
                inferred_dims = "CYX"
            else:
                data = data[np.newaxis, np.newaxis, :, :, :]
                ch_names = ["Channel 1"]
                inferred_dims = "ZYX"
        elif data.ndim == 4:
            if data.shape[-1] in (3, 4):
                rgb_stack = data[..., :3]
                data = np.moveaxis(rgb_stack, -1, 0)[np.newaxis, ...]
                ch_names = ["Red", "Green", "Blue"]
                inferred_dims = "CZYX"
            else:
                raise ValueError(
                    f"Unsupported image shape {data.shape}; expected grayscale stack or RGB(A) image/stack."
                )
        else:
            raise ValueError(
                f"Unsupported image shape {data.shape}; only 2D images, 3D stacks, and RGB(A) image stacks are supported."
            )

    meta: dict[str, Any] = {
        # Keep the returned array normalized to TCZYX internally, but expose the
        # inferred user-facing axes so downstream UI logic can distinguish ZYX
        # from TYX/TZYX correctly.
        "dims": inferred_dims or "TCZYX",
        "channel_names": ch_names,
        "unit": unit,
        # Keep napari's T coordinate in frame units.  The physical acquisition
        # interval remains available in time_interval/finterval/fps metadata.
        "scale_per_axis": (1.0, 1.0, *zyx_scale),
        "source": path,
        "axes": axes,
        "inferred_dims": inferred_dims,
        "storage_dims": "TCZYX",
    }
    meta.update({k: v for k, v in extras.items() if v is not None})
    return data, meta
