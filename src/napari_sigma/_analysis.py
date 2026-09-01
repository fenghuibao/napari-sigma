from __future__ import annotations

import csv
import os
import tempfile
from contextlib import suppress
from typing import Any

import numpy as np
from qtpy.QtCore import QItemSelectionModel, Qt

# Resolve once at import time; np.trapezoid was added in NumPy 2.0 and
# np.trapz was removed in NumPy 2.5. Avoid evaluating the legacy fallback
# unless it is actually needed.
if hasattr(np, "trapezoid"):
    _np_trapz = np.trapezoid
else:
    _np_trapz = np.trapz
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage as ndi
from skimage.graph import MCP
from skimage.measure import (
    label,
    perimeter,
    regionprops,
)
from skimage.morphology import skeletonize

from ._layer_candidates import limited_unique_values
from ._tracking import _label_frame

_MPL_CONFIG_DIR = os.path.join(tempfile.gettempdir(), "napari-sigma-matplotlib")
with suppress(OSError):
    os.makedirs(_MPL_CONFIG_DIR, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", _MPL_CONFIG_DIR)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
except (ImportError, RuntimeError, OSError):  # pragma: no cover
    Figure = None
    FigureCanvasQTAgg = None

try:
    from openpyxl import Workbook
except (ImportError, OSError):  # pragma: no cover
    Workbook = None


class AnalysisCancelledError(Exception):
    pass


def _get_units_from_layer(layer) -> str:
    md = getattr(layer, "metadata", {}) or {}
    return md.get("PhysicalSizeXUnit") or md.get("unit") or md.get("Units") or "um"


def _get_group_id(layer) -> str | None:
    md = getattr(layer, "metadata", {}) or {}
    gid = md.get("group_id")
    if gid:
        return str(gid)
    return None


def _layer_dims_tag(layer) -> str:
    md = getattr(layer, "metadata", {}) or {}
    return str(md.get("dims_out") or md.get("dims") or "").upper()


def _is_analysis_aux_layer(layer) -> bool:
    md = getattr(layer, "metadata", {}) or {}
    return bool(
        md.get("is_analysis_labels")
        or md.get("is_analysis_highlight")
        or md.get("is_analysis_topology")
    )


def _squeeze_leading_singletons(arr: np.ndarray, target_ndim: int) -> np.ndarray:
    out = np.asarray(arr)
    while out.ndim > target_ndim and out.shape[0] == 1:
        out = out[0]
    return out


def _component_mask_from_layer(layer, frame_index: int | None = None) -> np.ndarray:
    source = layer.data
    dims_tag = _layer_dims_tag(layer)
    if dims_tag == "TYX":
        if source.ndim < 3:
            raise ValueError(f"Unsupported TYX layer ndim for analysis: {source.ndim}")
        idx = 0 if frame_index is None else max(0, min(int(frame_index), source.shape[0] - 1))
        data = np.asarray(source[idx])
    elif dims_tag == "TZYX":
        if source.ndim < 4:
            raise ValueError(f"Unsupported TZYX layer ndim for analysis: {source.ndim}")
        idx = 0 if frame_index is None else max(0, min(int(frame_index), source.shape[0] - 1))
        data = np.asarray(source[idx])
    else:
        data = np.asarray(source)
        data = np.squeeze(data)
        if data.ndim > 3:
            data = _squeeze_leading_singletons(data, 3)
        data = np.squeeze(data)
    binary = np.asarray(data) > 0
    if binary.ndim not in {2, 3}:
        raise ValueError(f"Unsupported segmentation ndim for analysis: {binary.ndim}")
    return binary


def analyze_binary_components(
    binary: np.ndarray,
    *,
    unit: str,
    voxel_size: tuple[float, ...],
    frame_index: int | None = None,
    cancel_check=None,
) -> tuple[np.ndarray, list[dict[str, float | int]], dict[str, Any] | None, dict[int, dict[str, object]]]:
    if cancel_check is not None and cancel_check():
        raise AnalysisCancelledError("Analysis cancelled.")
    labels = _label_frame(np.asarray(binary))
    if labels.max() == 0:
        return labels, [], None, {}

    props = regionprops(labels)
    object_rows: list[dict[str, float | int]] = []
    topology_cache: dict[int, dict[str, object]] = {}

    if binary.ndim == 3:
        voxel_measure = float(voxel_size[0] * voxel_size[1] * voxel_size[2])
        size_name = "volume"
        size_unit = f"{unit}^3"
        count_name = "voxels"
    else:
        voxel_measure = float(voxel_size[0] * voxel_size[1])
        size_name = "area"
        size_unit = f"{unit}^2"
        count_name = "pixels"

    global_equivalent_radius: float | None = None
    terminal_prune_length: float | None = None
    if binary.ndim == 3:
        global_equivalent_radius = _global_equivalent_radius_3d(
            props,
            voxel_size,
            cancel_check=cancel_check,
        )
        if global_equivalent_radius is not None:
            terminal_prune_length = global_equivalent_radius

    for prop in props:
        if cancel_check is not None and cancel_check():
            raise AnalysisCancelledError("Analysis cancelled.")
        if int(prop.area) <= 1:
            continue
        size_value = float(prop.area * voxel_measure)
        topology = _skeleton_topology_details(
            prop.image,
            voxel_size,
            terminal_prune_length=terminal_prune_length,
        )
        fallback_length = _major_axis_length_in_units(prop, voxel_size)
        total_branch_length = float(topology["total_branch_length"])
        mean_branch_length = float(topology["mean_branch_length"])
        length_source = str(topology.get("length_source", "skeleton"))
        if total_branch_length <= 0.0 and fallback_length > 0.0:
            total_branch_length = fallback_length
            mean_branch_length = fallback_length
            length_source = "major_axis"
        branch_records = [dict(record) for record in topology.get("branch_records", ())]
        topology_cache[int(prop.label)] = {
            "slice": prop.slice,
            "skeleton_mask": np.asarray(topology["skeleton_mask"], dtype=bool),
            "endpoint_regions": [dict(region) for region in topology["endpoint_regions"]],
            "junction_regions": [dict(region) for region in topology["junction_regions"]],
            "branch_records": branch_records,
        }
        row: dict[str, float | int] = {
            "label": int(prop.label),
            count_name: int(prop.area),
            size_name: size_value,
            "total_branch_length": total_branch_length,
            "branch_number": int(topology["branch_number"]),
            "junction_number": int(topology["junction_number"]),
            "mean_branch_length": mean_branch_length,
            "endpoint_number": int(topology["endpoint_number"]),
            "length source": length_source,
        }
        if binary.ndim == 3:
            row["surface_area"] = _surface_area_in_units(np.asarray(prop.image, dtype=bool), voxel_size)
        else:
            row["perimeter"] = _perimeter_in_units(np.asarray(prop.image, dtype=bool), voxel_size)
        if frame_index is not None:
            row["frame"] = int(frame_index)
        object_rows.append(row)

    object_rows.sort(key=lambda row: float(row[size_name]), reverse=True)
    branch_length_values: list[float] = []
    for cached in topology_cache.values():
        for record in cached.get("branch_records", ()):
            with suppress(TypeError, ValueError):
                branch_length_values.append(float(record.get("length", 0.0)))

    plot_info = {
        "size_name": size_name,
        "size_unit": size_unit,
        "size_values": [float(row[size_name]) for row in object_rows],
        "branch_name": "branch number",
        "branch_unit": "",
        "branch_values": [float(row["branch_number"]) for row in object_rows],
        "branch_length_name": "branch length",
        "branch_length_unit": unit,
        "branch_length_values": branch_length_values,
        "count_name": count_name,
    }
    if frame_index is not None:
        plot_info["frame_index"] = int(frame_index)
    if global_equivalent_radius is not None and terminal_prune_length is not None:
        plot_info["global_equivalent_radius"] = float(global_equivalent_radius)
        plot_info["terminal_prune_length"] = float(terminal_prune_length)
    return labels, object_rows, plot_info, topology_cache


def write_analysis_export_file(
    path: str,
    headers: list[str],
    rows: list[list[str]],
    *,
    sheet_title: str = "measurements",
) -> None:
    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else "\t"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter=delimiter)
            writer.writerow(headers)
            writer.writerows(rows)
        return
    if suffix == ".xlsx":
        if Workbook is None:
            raise RuntimeError("openpyxl is not available in this environment.")
        wb = Workbook()
        ws = wb.active
        ws.title = str(sheet_title)[:31] or "measurements"
        ws.append(headers)
        for row in rows:
            ws.append(row)
        wb.save(path)
        return
    raise ValueError(f"Unsupported export format: {suffix}")


def _analysis_object_size_key(rows: list[dict[str, float | int]]) -> str | None:
    if not rows:
        return None
    sample = rows[0]
    if "voxels" in sample:
        return "voxels"
    if "pixels" in sample:
        return "pixels"
    return None


def _filter_analysis_rows_by_min_size(
    rows: list[dict[str, float | int]],
    min_size: int,
) -> list[dict[str, float | int]]:
    if not rows or int(min_size) <= 0:
        return list(rows)
    size_key = _analysis_object_size_key(rows)
    if size_key is None:
        return list(rows)
    threshold = int(min_size)
    return [row for row in rows if int(row.get(size_key, 0)) >= threshold]


def _analysis_branch_rows_from_rows(
    rows: list[dict[str, float | int]],
    topology_cache: dict[int, dict[str, object]],
    frame_index: int | None,
) -> list[dict[str, float | int]]:
    branch_rows: list[dict[str, float | int]] = []
    global_branch_index = 0
    for row in rows:
        object_label = int(row.get("label", 0))
        cached = topology_cache.get(object_label) or {}
        for object_branch_index, record in enumerate(cached.get("branch_records", ()), start=1):
            global_branch_index += 1
            branch_row: dict[str, float | int] = {
                "object_label": object_label,
                "branch_index": int(global_branch_index),
                "_object_branch_index": int(object_branch_index),
                "branch_length": float(record.get("length", 0.0) or 0.0),
            }
            if frame_index is not None:
                branch_row["frame"] = int(frame_index)
            branch_rows.append(branch_row)
    branch_rows.sort(key=lambda item: float(item.get("branch_length", 0.0)), reverse=True)
    return branch_rows


def _filter_plot_info_for_rows(
    plot_info: dict[str, Any] | None,
    rows: list[dict[str, float | int]],
    topology_cache: dict[int, dict[str, object]],
) -> dict[str, Any] | None:
    if plot_info is None:
        return None
    filtered = dict(plot_info)
    size_name = str(filtered.get("size_name") or "")
    branch_name = str(filtered.get("branch_name") or "")
    if size_name:
        filtered["size_values"] = [float(row.get(size_name, 0.0) or 0.0) for row in rows]
    if branch_name:
        filtered["branch_values"] = [float(row.get("branch_number", 0.0) or 0.0) for row in rows]
    filtered_labels = {int(row.get("label", 0)) for row in rows}
    branch_length_values: list[float] = []
    for label_id in sorted(filtered_labels):
        cached = topology_cache.get(label_id) or {}
        for record in cached.get("branch_records", ()):
            with suppress(TypeError, ValueError):
                value = float(record.get("length", 0.0) or 0.0)
                if value > 0:
                    branch_length_values.append(value)
    filtered["branch_length_values"] = branch_length_values
    return filtered


def _lorenz_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (x, cumulative) arrays for the Lorenz curve, or None if no valid values."""
    arr = np.sort(np.asarray(values, dtype=float))
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return None
    cum = np.cumsum(arr, dtype=float)
    cum = np.concatenate([[0.0], cum / cum[-1]])
    return np.linspace(0.0, 1.0, cum.size), cum


def _style_ax(ax, fg: str, grid: str) -> None:
    """Apply consistent dark-theme colours to a matplotlib Axes."""
    ax.tick_params(colors=fg)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)
    for spine in ax.spines.values():
        spine.set_color(grid)


def _make_table_item(value: float | int) -> "QTableWidgetItem":
    """Create a centered, read-only QTableWidgetItem formatted for display."""
    item = QTableWidgetItem(f"{value:.6g}" if isinstance(value, float) else str(value))
    item.setTextAlignment(Qt.AlignCenter)
    return item


def _spatial_scale_for_ndim(layer, ndim: int) -> tuple[float, ...] | None:
    sc = getattr(layer, "scale", None)
    if sc is None:
        return None
    sc_tuple = tuple(float(v) for v in sc)
    if len(sc_tuple) >= ndim:
        return sc_tuple[-ndim:]
    return None


def _spatial_units_for_ndim(layer, ndim: int) -> tuple[object, ...] | None:
    units = getattr(layer, "units", None)
    if units is None:
        return None
    units_tuple = tuple(units)
    if len(units_tuple) >= ndim:
        return units_tuple[-ndim:]
    return None


def _get_pixel_size_tuple(layer, ndim: int):
    sc = getattr(layer, "scale", (1,) * layer.data.ndim)
    if ndim == 3:
        return (float(sc[-3]), float(sc[-2]), float(sc[-1]))
    if ndim == 2:
        return (float(sc[-2]), float(sc[-1]))
    raise ValueError(f"Unsupported ndim for pixel size: {ndim}")


def _is_analysis_candidate_layer(layer) -> bool:
    md = getattr(layer, "metadata", {}) or {}
    if md.get("is_analysis_labels") or _is_analysis_aux_layer(layer):
        return False
    layer_type = str(getattr(layer, "_type_string", "") or "")
    if layer_type not in {"image", "labels"}:
        return False
    if md.get("is_segmentation"):
        return True
    data = getattr(layer, "data", None)
    if data is None or int(getattr(data, "size", 0)) == 0:
        return False
    dims_tag = _layer_dims_tag(layer)
    if dims_tag == "TYX":
        if data.ndim < 3 or int(data.shape[0]) <= 0:
            return False
        data = np.asarray(data[0])
    elif dims_tag == "TZYX":
        if data.ndim < 4 or int(data.shape[0]) <= 0:
            return False
        data = np.asarray(data[0])
    else:
        shape = tuple(int(value) for value in getattr(data, "shape", ()))
        squeeze_index = tuple(0 if value == 1 else slice(None) for value in shape)
        if any(value == 1 for value in shape):
            data = data[squeeze_index]
        if getattr(data, "ndim", None) not in {2, 3}:
            return False
    unique = limited_unique_values(data, 16)
    return bool(unique is not None and np.all(np.isfinite(unique)))


def _is_related_group_layer(layer_item, layer, group_id: str | None) -> bool:
    md = getattr(layer_item, "metadata", {}) or {}
    same_group = group_id is not None and md.get("group_id") == group_id
    return bool(
        layer_item is layer
        or md.get("is_frangi")
        or md.get("is_denoise")
        or (md.get("is_segmentation") and same_group)
        or same_group
    )


def _neighbor_offsets(ndim: int) -> list[tuple[int, ...]]:
    if ndim == 2:
        return [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    if ndim == 3:
        offsets = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == dy == dx == 0:
                        continue
                    offsets.append((dz, dy, dx))
        return offsets
    raise ValueError(f"Unsupported ndim for skeleton neighbors: {ndim}")


def _filtered_neighbors(point: tuple[int, ...], points: set[tuple[int, ...]], ndim: int) -> list[tuple[int, ...]]:
    neighbors: list[tuple[int, ...]] = []
    for offset in _neighbor_offsets(ndim):
        candidate = tuple(point[i] + offset[i] for i in range(ndim))
        if candidate not in points:
            continue
        if ndim == 2 and offset[0] != 0 and offset[1] != 0:
            bridge_a = (point[0] + offset[0], point[1])
            bridge_b = (point[0], point[1] + offset[1])
            if bridge_a in points or bridge_b in points:
                continue
        if ndim == 3 and sum(int(v != 0) for v in offset) >= 2:
            axes = [axis for axis, delta in enumerate(offset) if delta != 0]
            bridge_points: list[tuple[int, int, int]] = []
            if len(axes) == 2:
                a0, a1 = axes
                bp1 = list(point)
                bp1[a0] += offset[a0]
                bp2 = list(point)
                bp2[a1] += offset[a1]
                bridge_points = [tuple(bp1), tuple(bp2)]
            elif len(axes) == 3:
                for keep_axis in axes:
                    bp_single = list(point)
                    bp_single[keep_axis] += offset[keep_axis]
                    bridge_points.append(tuple(bp_single))

                    bp_double = list(point)
                    for axis in axes:
                        if axis != keep_axis:
                            bp_double[axis] += offset[axis]
                    bridge_points.append(tuple(bp_double))
            if any(bp in points for bp in bridge_points):
                continue
        neighbors.append(candidate)
    return neighbors


def _step_length(a: tuple[int, ...], b: tuple[int, ...], pixel_size: tuple[float, ...]) -> float:
    delta = np.subtract(b, a, dtype=float)
    scale = np.asarray(pixel_size, dtype=float)
    return float(np.sqrt(np.sum((delta * scale) ** 2)))


def _map_isotropic_point_to_original(
    coord: tuple[int, int, int] | np.ndarray,
    original_mask: np.ndarray,
    zoom_factors: tuple[float, float, float],
) -> tuple[int, int, int] | None:
    shape = np.asarray(original_mask.shape, dtype=int)
    mapped = np.rint(np.asarray(coord, dtype=float) / np.asarray(zoom_factors, dtype=float)).astype(int)
    mapped = np.clip(mapped, 0, shape - 1)
    point = tuple(int(v) for v in mapped.tolist())
    if original_mask[point]:
        return point
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                candidate = (point[0] + dz, point[1] + dy, point[2] + dx)
                if any(candidate[i] < 0 or candidate[i] >= shape[i] for i in range(3)):
                    continue
                if original_mask[candidate]:
                    return candidate
    return None


def _project_isotropic_skeleton_to_mask(
    isotropic_skeleton: np.ndarray,
    original_mask: np.ndarray,
    zoom_factors: tuple[float, float, float],
) -> np.ndarray:
    projected = np.zeros_like(original_mask, dtype=bool)
    coords = np.argwhere(isotropic_skeleton > 0)
    if coords.size == 0:
        return projected
    coord_tuples = {tuple(int(v) for v in coord) for coord in coords}
    mapped_points: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
    for coord in coord_tuples:
        point = _map_isotropic_point_to_original(np.asarray(coord), original_mask, zoom_factors)
        mapped_points[coord] = point
        if point is not None:
            projected[point] = True
    for coord in coord_tuples:
        src = mapped_points.get(coord)
        if src is None:
            continue
        for neighbor in _filtered_neighbors(coord, coord_tuples, 3):
            if neighbor <= coord:
                continue
            dst = mapped_points.get(neighbor)
            if dst is None or dst == src:
                continue
            for point in _sample_line_points_3d(src, dst):
                point_arr = np.asarray(point)
                if np.any(point_arr < 0) or np.any(point_arr >= np.asarray(original_mask.shape)):
                    continue
                if original_mask[point]:
                    projected[point] = True
    return projected


def _project_isotropic_regions_to_mask(
    regions: list[dict[str, Any]],
    original_mask: np.ndarray,
    zoom_factors: tuple[float, float, float],
) -> list[dict[str, Any]]:
    projected_regions: list[dict[str, Any]] = []
    for region in regions:
        mapped_points: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        for point in region.get("points", ()):
            mapped = _map_isotropic_point_to_original(point, original_mask, zoom_factors)
            if mapped is None or mapped in seen:
                continue
            seen.add(mapped)
            mapped_points.append(mapped)
        if not mapped_points:
            center = region.get("center")
            if center is not None:
                mapped = _map_isotropic_point_to_original(tuple(int(round(v)) for v in center), original_mask, zoom_factors)
                if mapped is not None:
                    mapped_points = [mapped]
        if not mapped_points:
            continue
        coords = np.asarray(mapped_points, dtype=float)
        projected_regions.append(
            {
                "center": tuple(float(v) for v in coords.mean(axis=0)),
                "size": int(len(mapped_points)),
                "points": mapped_points,
            }
        )
    return projected_regions


def _project_isotropic_branch_records_to_mask(
    branch_records: list[dict[str, Any]],
    original_mask: np.ndarray,
    zoom_factors: tuple[float, float, float],
) -> list[dict[str, Any]]:
    projected_records: list[dict[str, Any]] = []
    for record in branch_records:
        mapped_points: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        for point in record.get("points", ()):
            mapped = _map_isotropic_point_to_original(point, original_mask, zoom_factors)
            if mapped is None or mapped in seen:
                continue
            seen.add(mapped)
            mapped_points.append(mapped)
        projected_length = 0.0
        for start, end in zip(mapped_points[:-1], mapped_points[1:], strict=False):
            projected_length += _step_length(start, end, (1.0, 1.0, 1.0))
        projected_record = {
            "start_node": record.get("start_node"),
            "end_node": record.get("end_node"),
            "length": float(record.get("length", projected_length) or projected_length),
            "points": [tuple(point) for point in mapped_points],
        }
        if len(mapped_points) < 2:
            projected_record["_projection_degenerate"] = True
        projected_records.append(projected_record)
    return projected_records


def _skeletonize_lee(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim == 2:
        return skeletonize(binary)
    if binary.ndim == 3:
        return skeletonize(binary, method="lee")
    raise ValueError(f"Unsupported ndim for topology analysis: {binary.ndim}")


def _regions_by_id(labels: np.ndarray) -> dict[int, dict[str, Any]]:
    if not np.any(labels):
        return {}
    return {
        int(region.label): {
            "center": tuple(float(v) for v in region.centroid),
            "size": int(region.area),
            "points": [tuple(int(v) for v in point) for point in region.coords],
        }
        for region in regionprops(labels)
    }


def _principal_axis_fallback_topology(
    skeleton_mask: np.ndarray,
    pixel_size: tuple[float, ...],
) -> dict[str, Any] | None:
    coords = np.argwhere(skeleton_mask > 0)
    if coords.size == 0:
        return None
    if len(coords) == 1:
        center = tuple(float(v) for v in coords[0])
        return {
            "length_source": "principal_axis",
            "total_branch_length": 0.0,
            "branch_number": 0,
            "junction_number": 0,
            "mean_branch_length": 0.0,
            "endpoint_number": 1,
            "skeleton_mask": skeleton_mask.astype(bool),
            "junction_regions": [],
            "endpoint_regions": [{"center": center, "size": 1}],
            "branch_records": [],
        }

    coords_f = coords.astype(float)
    centered = coords_f - coords_f.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projections = centered @ axis
    p0 = tuple(int(v) for v in coords[int(np.argmin(projections))])
    p1 = tuple(int(v) for v in coords[int(np.argmax(projections))])
    line_points = [point for point in _sample_line_points_3d(p0, p1) if skeleton_mask[point]]
    if not line_points:
        line_points = [p0, p1] if p0 != p1 else [p0]
    axis_mask = np.zeros_like(skeleton_mask, dtype=bool)
    for point in line_points:
        axis_mask[point] = True
    branch_length = float(_step_length(p0, p1, pixel_size)) if p0 != p1 else 0.0
    endpoint_regions = [
        {"center": tuple(float(v) for v in p0), "size": 1},
        {"center": tuple(float(v) for v in p1), "size": 1},
    ]
    if p0 == p1:
        endpoint_regions = endpoint_regions[:1]
    branch_records = []
    if p0 != p1:
        branch_records.append(
            {
                "start_node": ("endpoint", 1),
                "end_node": ("endpoint", 2),
                "length": branch_length,
                "points": [tuple(point) for point in line_points],
            }
        )
    return {
        "length_source": "principal_axis",
        "total_branch_length": branch_length,
        "branch_number": int(1 if p0 != p1 else 0),
        "junction_number": 0,
        "mean_branch_length": branch_length,
        "endpoint_number": int(len(endpoint_regions)),
        "skeleton_mask": axis_mask,
        "junction_regions": [],
        "endpoint_regions": endpoint_regions,
        "branch_records": branch_records,
    }


def _needs_principal_axis_fallback(topology: dict[str, Any]) -> bool:
    branch_records = list(topology.get("branch_records", ()))
    missing_projected_axis = (
        str(topology.get("length_source", "")) == "principal_axis"
        and int(topology.get("branch_number", 0)) > 0
        and (
            len(branch_records) != int(topology.get("branch_number", 0))
            or any(len(record.get("points", ())) < 2 for record in branch_records)
        )
    )
    return missing_projected_axis or (
        int(topology.get("branch_number", 0)) == 0
        and int(topology.get("junction_number", 0)) == 0
        and int(topology.get("endpoint_number", 0)) <= 1
    )


def _empty_topology_details(mask: np.ndarray) -> dict[str, Any]:
    return {
        "length_source": "skeleton",
        "total_branch_length": 0.0,
        "branch_number": 0,
        "junction_number": 0,
        "mean_branch_length": 0.0,
        "endpoint_number": 0,
        "skeleton_mask": np.zeros_like(mask, dtype=bool),
        "junction_regions": [],
        "endpoint_regions": [],
    }


def _project_isotropic_topology_details(
    iso_details: dict[str, Any],
    mask: np.ndarray,
    zoom_factors: tuple[float, float, float],
) -> dict[str, Any]:
    mask_bool = np.asarray(mask, dtype=bool)
    return {
        "length_source": str(iso_details.get("length_source", "skeleton")),
        "total_branch_length": float(iso_details["total_branch_length"]),
        "branch_number": int(iso_details["branch_number"]),
        "junction_number": int(iso_details["junction_number"]),
        "mean_branch_length": float(iso_details["mean_branch_length"]),
        "endpoint_number": int(iso_details["endpoint_number"]),
        "skeleton_mask": _project_isotropic_skeleton_to_mask(
            np.asarray(iso_details["skeleton_mask"], dtype=bool),
            mask_bool,
            zoom_factors,
        ),
        "junction_regions": _project_isotropic_regions_to_mask(
            list(iso_details["junction_regions"]),
            mask_bool,
            zoom_factors,
        ),
        "endpoint_regions": _project_isotropic_regions_to_mask(
            list(iso_details["endpoint_regions"]),
            mask_bool,
            zoom_factors,
        ),
        "branch_records": _project_isotropic_branch_records_to_mask(
            list(iso_details.get("branch_records", [])),
            mask_bool,
            zoom_factors,
        ),
    }


def _skeletonize_object_isotropic(
    mask: np.ndarray,
    pixel_size: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
    binary = np.asarray(mask, dtype=bool)
    scale = np.asarray(pixel_size, dtype=float)
    target = float(np.min(scale))
    zoom_factors = tuple(float(v / target) for v in scale)
    iso_mask_raw = ndi.zoom(binary.astype(np.uint8), zoom_factors, order=0) > 0
    iso_mask = _smooth_binary_object_mask_3d(iso_mask_raw)
    iso_skeleton = _skeletonize_lee(iso_mask)
    iso_pixel_size = (target, target, target)
    iso_skeleton = _bridge_nearby_endpoints_3d(
        np.asarray(iso_skeleton, dtype=bool),
        np.asarray(iso_mask_raw, dtype=bool),
    )
    return iso_mask, iso_skeleton.astype(bool), zoom_factors, iso_pixel_size


def _skeleton_graph_length(
    skeleton: np.ndarray,
    pixel_size: tuple[float, ...],
) -> float:
    points = {tuple(coord.tolist()) for coord in np.argwhere(skeleton > 0)}
    if len(points) < 2:
        return 0.0

    visited_edges: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    total_length = 0.0
    for point in points:
        for neighbor in _filtered_neighbors(point, points, skeleton.ndim):
            edge = tuple(sorted((point, neighbor)))
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            total_length += _step_length(point, neighbor, pixel_size)
    return float(total_length)


def _global_equivalent_radius_3d(
    props,
    pixel_size: tuple[float, ...],
    *,
    cancel_check=None,
) -> float | None:
    if len(pixel_size) != 3:
        return None

    voxel_volume = float(np.prod(np.asarray(pixel_size, dtype=float)))
    total_volume = 0.0
    total_length = 0.0
    for prop in props:
        if cancel_check is not None and cancel_check():
            raise AnalysisCancelledError("Analysis cancelled.")
        if int(prop.area) <= 1:
            continue

        total_volume += float(prop.area) * voxel_volume
        binary = np.asarray(prop.image, dtype=bool)
        scale = np.asarray(pixel_size, dtype=float)
        target = float(np.min(scale))
        zoom_factors = tuple(float(value / target) for value in scale)
        iso_mask_raw = ndi.zoom(binary.astype(np.uint8), zoom_factors, order=0) > 0
        iso_mask = _smooth_binary_object_mask_3d(iso_mask_raw)
        iso_pixel_size = (target, target, target)
        lee_skeleton = _skeletonize_lee(iso_mask)
        object_length = _skeleton_graph_length(lee_skeleton, iso_pixel_size)
        if not np.isfinite(object_length) or object_length <= 0.0:
            object_length = _major_axis_length_in_units(prop, pixel_size)
        if np.isfinite(object_length) and object_length > 0.0:
            total_length += float(object_length)

    if total_volume <= 0.0 or total_length <= 0.0:
        return None
    return float(np.sqrt(total_volume / (np.pi * total_length)))


def _smooth_binary_object_mask_3d(mask: np.ndarray, sigma: float = 0.75, threshold: float = 0.5) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3 or not np.any(binary):
        return binary
    smoothed = ndi.gaussian_filter(binary.astype(np.float32), sigma=float(sigma), mode="nearest")
    return smoothed >= float(threshold)


def _sample_line_points_3d(a: tuple[float, float, float], b: tuple[float, float, float]) -> list[tuple[int, int, int]]:
    start = np.asarray(a, dtype=float)
    end = np.asarray(b, dtype=float)
    delta = end - start
    steps = int(np.ceil(np.max(np.abs(delta))))
    if steps <= 0:
        point = tuple(np.rint(start).astype(int).tolist())
        return [point]
    points: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for t in np.linspace(0.0, 1.0, steps + 1):
        point = tuple(np.rint(start + t * delta).astype(int).tolist())
        if point in seen:
            continue
        seen.add(point)
        points.append(point)
    return points


def _skeleton_component_labels(skeleton: np.ndarray) -> tuple[np.ndarray, int]:
    binary = np.asarray(skeleton, dtype=bool)
    if binary.ndim == 3:
        structure = ndi.generate_binary_structure(3, 3)
        labels, count = ndi.label(binary, structure=structure)
        return labels, int(count)
    labels, count = ndi.label(binary)
    return labels, int(count)


def _bridge_nearby_endpoints_3d(
    skeleton: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    out = np.asarray(skeleton, dtype=bool).copy()
    if out.ndim != 3:
        return out
    mask_bool = np.asarray(mask, dtype=bool)
    component_labels, component_count = _skeleton_component_labels(out)
    if component_count <= 1:
        return out
    component_points: dict[int, list[tuple[int, int, int]]] = {}
    for point in map(tuple, np.argwhere(out)):
        comp_id = int(component_labels[point])
        if comp_id > 0:
            component_points.setdefault(comp_id, []).append(
                tuple(int(value) for value in point)
            )
    if len(component_points) <= 1:
        return out

    largest_comp = max(component_points, key=lambda cid: len(component_points[cid]))
    costs = np.where(mask_bool, 1.0, np.inf)
    path_finder = MCP(costs, fully_connected=True)
    cumulative_costs, _ = path_finder.find_costs(
        starts=component_points[largest_comp],
    )
    for comp_id, points in component_points.items():
        if comp_id == largest_comp or not points:
            continue
        point_array = np.asarray(points, dtype=int)
        point_costs = cumulative_costs[tuple(point_array.T)]
        reachable = np.flatnonzero(np.isfinite(point_costs))
        if reachable.size == 0:
            continue
        best_index = int(reachable[np.argmin(point_costs[reachable])])
        best_end = points[best_index]
        for point in path_finder.traceback(best_end):
            out[tuple(int(value) for value in point)] = True
    return out


def _topology_details_from_skeleton(
    skeleton: np.ndarray,
    pixel_size: tuple[float, ...],
    mask: np.ndarray | None = None,
    terminal_prune_length: float | None = None,
) -> dict[str, Any]:
    branch_records, endpoint_regions_by_id, junction_regions_by_id = _extract_branch_records_from_skeleton(
        skeleton,
        pixel_size,
    )
    if skeleton.ndim == 3 and branch_records:
        branch_records = _prune_short_terminal_branches_3d(
            branch_records,
            pixel_size,
            mask=mask,
            junction_regions_by_id=junction_regions_by_id,
            terminal_prune_length=terminal_prune_length,
        )

    simplified_skeleton = np.zeros_like(skeleton, dtype=bool)
    for branch in branch_records:
        for point in branch["points"]:
            simplified_skeleton[point] = True
        start_node = branch.get("start_node")
        if start_node is not None:
            region = endpoint_regions_by_id.get(start_node[1]) if start_node[0] == "endpoint" else junction_regions_by_id.get(start_node[1])
            if region is not None:
                for point in region.get("points", ()):
                    simplified_skeleton[point] = True
        end_node = branch.get("end_node")
        if end_node is not None:
            region = endpoint_regions_by_id.get(end_node[1]) if end_node[0] == "endpoint" else junction_regions_by_id.get(end_node[1])
            if region is not None:
                for point in region.get("points", ()):
                    simplified_skeleton[point] = True
    if not np.any(simplified_skeleton):
        simplified_skeleton = skeleton.astype(bool)

    final_branch_records, endpoint_regions_by_id, junction_regions_by_id = _extract_branch_records_from_skeleton(
        simplified_skeleton,
        pixel_size,
    )
    if skeleton.ndim == 3 and final_branch_records:
        final_branch_records = _prune_short_terminal_branches_3d(
            final_branch_records,
            pixel_size,
            mask=mask,
            junction_regions_by_id=junction_regions_by_id,
            terminal_prune_length=terminal_prune_length,
        )
        previous_count = -1
        while previous_count != len(final_branch_records):
            previous_count = len(final_branch_records)
            if mask is not None:
                final_branch_records = _drop_false_loops_3d(final_branch_records, mask)
            final_branch_records = _contract_degree_two_junctions(final_branch_records)

    node_branch_counts: dict[tuple[str, int], int] = {}
    for branch in final_branch_records:
        start_node = branch.get("start_node")
        end_node = branch.get("end_node")
        if start_node is not None:
            node_branch_counts[start_node] = node_branch_counts.get(start_node, 0) + 1
        if end_node is not None:
            node_branch_counts[end_node] = node_branch_counts.get(end_node, 0) + 1

    branch_lengths = [float(branch["length"]) for branch in final_branch_records]
    branch_number = len(branch_lengths)
    total_branch_length = float(np.sum(branch_lengths)) if branch_lengths else 0.0
    mean_branch_length = float(total_branch_length / branch_number) if branch_number > 0 else 0.0
    endpoint_regions: list[dict[str, Any]] = []
    for (node_type, node_id), degree in node_branch_counts.items():
        if degree < 1:
            continue
        if node_type == "endpoint" and degree == 1 and node_id in endpoint_regions_by_id:
            endpoint_regions.append(endpoint_regions_by_id[node_id])
        elif node_type == "junction" and degree == 1 and node_id in junction_regions_by_id:
            endpoint_regions.append(junction_regions_by_id[node_id])
    endpoint_points = {
        tuple(int(round(v)) for v in region["center"])
        for region in endpoint_regions
        if "center" in region
    }
    for branch in final_branch_records:
        start_node = branch.get("start_node")
        end_node = branch.get("end_node")
        if not ((start_node is None) ^ (end_node is None)):
            continue
        points = [tuple(point) for point in branch.get("points", ())]
        if not points:
            continue
        tip = points[0] if start_node is None else points[-1]
        if tip in endpoint_points:
            continue
        endpoint_points.add(tip)
        endpoint_regions.append({"center": tuple(float(v) for v in tip), "size": 1})
    if not endpoint_regions:
        fallback = _principal_axis_fallback_topology(simplified_skeleton, pixel_size)
        if fallback is not None:
            return fallback
    endpoint_number = len(endpoint_regions)
    junction_regions = [
        junction_regions_by_id[node_id]
        for (node_type, node_id), degree in node_branch_counts.items()
        if node_type == "junction" and degree >= 3 and node_id in junction_regions_by_id
    ]
    return {
        "length_source": "skeleton",
        "total_branch_length": total_branch_length,
        "branch_number": int(branch_number),
        "junction_number": int(len(junction_regions)),
        "mean_branch_length": mean_branch_length,
        "endpoint_number": int(endpoint_number),
        "skeleton_mask": simplified_skeleton,
        "junction_regions": junction_regions,
        "endpoint_regions": endpoint_regions,
        "branch_records": final_branch_records,
    }


def _loop_center_has_background(
    points: list[tuple[int, ...]],
    mask: np.ndarray,
) -> bool:
    if not points:
        return False
    coords = np.asarray(points, dtype=int)
    mins = np.maximum(coords.min(axis=0) - 1, 0)
    maxs = np.minimum(coords.max(axis=0) + 2, np.asarray(mask.shape, dtype=int))
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins.tolist(), maxs.tolist(), strict=False))
    local_mask = np.asarray(mask[slices], dtype=bool)
    if local_mask.ndim == 3:
        for axis in range(3):
            proj_mask = np.any(local_mask, axis=axis)
            filled_mask = ndi.binary_fill_holes(proj_mask)
            if np.any(filled_mask & ~proj_mask):
                return True
        return False
    filled_mask = ndi.binary_fill_holes(local_mask)
    return bool(np.any(filled_mask & ~local_mask))


def _drop_false_parallel_loops(
    branch_records: list[dict[str, Any]],
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[object, object], list[dict[str, Any]]] = {}
    for branch in branch_records:
        key = tuple(sorted((branch.get("start_node"), branch.get("end_node")), key=repr))
        grouped.setdefault(key, []).append(branch)
    kept: list[dict[str, Any]] = []
    for key, branches in grouped.items():
        nodes = [node for node in key if node is not None]
        if len(branches) <= 1 or len(nodes) != 2:
            kept.extend(branches)
            continue
        loop_points: list[tuple[int, ...]] = []
        for branch in branches:
            loop_points.extend(branch.get("points", []))
        if _loop_center_has_background(loop_points, mask):
            kept.extend(branches)
            continue
        kept.extend(
            sorted(
                branches,
                key=lambda branch: (
                    float(branch.get("length", 0.0)),
                    len(branch.get("points", [])),
                ),
            )[:1]
        )
    return kept


def _drop_false_loops_3d(
    branch_records: list[dict[str, Any]],
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    if not branch_records:
        return branch_records
    kept: list[dict[str, Any]] = []
    for branch in branch_records:
        start_node = branch.get("start_node")
        end_node = branch.get("end_node")
        if start_node is not None and start_node == end_node:
            loop_points = [tuple(point) for point in branch.get("points", ())]
            if _loop_center_has_background(loop_points, mask):
                kept.append(branch)
            continue
        kept.append(branch)
    return _drop_false_parallel_loops(kept, mask)


def _contract_degree_two_junctions(
    branch_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = [dict(branch) for branch in branch_records]
    while True:
        incidences: dict[tuple[str, int], list[tuple[int, str]]] = {}
        for index, branch in enumerate(records):
            start_node = branch.get("start_node")
            end_node = branch.get("end_node")
            if start_node is not None:
                incidences.setdefault(start_node, []).append((index, "start"))
            if end_node is not None:
                incidences.setdefault(end_node, []).append((index, "end"))

        candidate = None
        for node in sorted(incidences, key=repr):
            incident = incidences[node]
            if (
                node[0] == "junction"
                and len(incident) == 2
                and incident[0][0] != incident[1][0]
            ):
                candidate = (node, incident[0][0], incident[1][0])
                break
        if candidate is None:
            return records

        node, first_index, second_index = candidate
        first = records[first_index]
        second = records[second_index]

        def _oriented_to_node(branch: dict[str, Any]) -> tuple[object, list[tuple[int, ...]]]:
            points = [tuple(point) for point in branch.get("points", ())]
            if branch.get("end_node") == node:
                return branch.get("start_node"), points
            if branch.get("start_node") == node:
                return branch.get("end_node"), list(reversed(points))
            raise ValueError(f"Branch is not incident to {node!r}")

        first_other, first_points = _oriented_to_node(first)
        second_other, second_points_to_node = _oriented_to_node(second)
        second_points = list(reversed(second_points_to_node))
        if first_points and second_points and first_points[-1] == second_points[0]:
            merged_points = first_points + second_points[1:]
        else:
            merged_points = first_points + second_points
        merged = {
            "start_node": first_other,
            "end_node": second_other,
            "length": float(first.get("length", 0.0)) + float(second.get("length", 0.0)),
            "points": merged_points,
        }

        for index in sorted((first_index, second_index), reverse=True):
            records.pop(index)
        records.append(merged)


def _prune_short_terminal_branches_3d(
    branch_records: list[dict[str, Any]],
    pixel_size: tuple[float, ...],
    *,
    mask: np.ndarray | None = None,
    junction_regions_by_id: dict[int, dict[str, Any]] | None = None,
    terminal_prune_length: float | None = None,
) -> list[dict[str, Any]]:
    if not branch_records or len(pixel_size) != 3:
        return branch_records

    # Full-image analyses provide one equivalent-diameter threshold for every
    # object. The local-radius path remains for direct single-object callers.
    min_terminal_length = 2.0 * float(np.mean(pixel_size[-2:]))
    uniform_terminal_length: float | None = None
    if terminal_prune_length is not None:
        candidate = float(terminal_prune_length)
        if np.isfinite(candidate) and candidate >= 0.0:
            uniform_terminal_length = candidate

    junction_min_lengths: dict[int, float] = {}
    if uniform_terminal_length is None and mask is not None and junction_regions_by_id:
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.ndim == 3 and mask_bool.shape:
            distance = ndi.distance_transform_edt(mask_bool, sampling=pixel_size)
            for junction_id, region in junction_regions_by_id.items():
                radii = []
                for point in region.get("points", ()):
                    point_tuple = tuple(int(value) for value in point)
                    if (
                        len(point_tuple) == 3
                        and all(0 <= point_tuple[axis] < mask_bool.shape[axis] for axis in range(3))
                        and mask_bool[point_tuple]
                    ):
                        radii.append(float(distance[point_tuple]))
                if radii:
                    junction_min_lengths[int(junction_id)] = max(
                        min_terminal_length,
                        2.0 * float(np.max(radii)),
                    )

    def _junction_threshold(node) -> float:
        if uniform_terminal_length is not None:
            return uniform_terminal_length
        if node is None or node[0] != "junction":
            return min_terminal_length
        return junction_min_lengths.get(int(node[1]), min_terminal_length)

    kept: list[dict[str, Any]] = []
    for branch in branch_records:
        start_node = branch.get("start_node")
        end_node = branch.get("end_node")
        if (start_node is None) ^ (end_node is None):
            live_node = start_node if start_node is not None else end_node
            if (
                live_node is not None
                and live_node[0] == "junction"
                and float(branch.get("length", 0.0)) <= _junction_threshold(live_node)
            ):
                continue
            kept.append(branch)
            continue
        if start_node is None or end_node is None:
            kept.append(branch)
            continue
        node_types = {start_node[0], end_node[0]}
        junction_node = start_node if start_node[0] == "junction" else end_node
        if (
            node_types == {"endpoint", "junction"}
            and float(branch.get("length", 0.0)) <= _junction_threshold(junction_node)
        ):
            continue
        kept.append(branch)
    return kept


def _dedupe_branch_records(
    branch_records: list[dict[str, Any]],
    node_lookup,
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[object, ...]] = set()
    for branch in branch_records:
        start_node = branch.get("start_node")
        end_node = branch.get("end_node")
        node_pair = tuple(sorted((start_node, end_node), key=repr))
        path_core = tuple(
            sorted(
                point
                for point in branch.get("points", [])
                if node_lookup(point) is None
            )
        )
        signature = (node_pair, path_core)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(branch)
    return deduped


def _extract_branch_records_from_skeleton(
    skeleton: np.ndarray,
    pixel_size: tuple[float, ...],
) -> tuple[
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    coords = np.argwhere(skeleton > 0)
    if coords.size == 0:
        return [], {}, {}

    points = {tuple(coord.tolist()) for coord in coords}
    neighbor_map = {point: _filtered_neighbors(point, points, skeleton.ndim) for point in points}
    degrees = {point: len(neighbors) for point, neighbors in neighbor_map.items()}
    endpoint_mask = np.zeros_like(skeleton, dtype=np.uint8)
    junction_mask = np.zeros_like(skeleton, dtype=np.uint8)
    for point, degree in degrees.items():
        if degree == 1:
            endpoint_mask[point] = 1
        elif degree >= 3:
            junction_mask[point] = 1

    has_endpoints = bool(np.any(endpoint_mask))
    has_junctions = bool(np.any(junction_mask))
    endpoint_labels = label(endpoint_mask, connectivity=skeleton.ndim) if has_endpoints else np.zeros_like(endpoint_mask)
    junction_labels = label(junction_mask, connectivity=skeleton.ndim) if has_junctions else np.zeros_like(junction_mask)

    def node_key(point: tuple[int, ...]) -> tuple[str, int] | None:
        endpoint_id = int(endpoint_labels[point]) if has_endpoints else 0
        if endpoint_id > 0:
            return ("endpoint", endpoint_id)
        junction_id = int(junction_labels[point]) if has_junctions else 0
        if junction_id > 0:
            return ("junction", junction_id)
        return None

    def edge_key(p0: tuple[int, ...], p1: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return tuple(sorted((p0, p1)))

    branch_records: list[dict[str, Any]] = []
    visited_edges: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    node_region_points: dict[tuple[str, int], set[tuple[int, ...]]] = {}
    for point in points:
        nk = node_key(point)
        if nk is not None:
            node_region_points.setdefault(nk, set()).add(point)

    for start_node, start_region_points in node_region_points.items():
        boundary_edges: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for start_point in start_region_points:
            for neighbor in neighbor_map[start_point]:
                if node_key(neighbor) == start_node:
                    continue
                boundary_edges.append((start_point, neighbor))
        seen_neighbors: set[tuple[int, ...]] = set()
        for start_point, neighbor in boundary_edges:
            if neighbor in seen_neighbors:
                continue
            seen_neighbors.add(neighbor)
            ek = edge_key(start_point, neighbor)
            if ek in visited_edges:
                continue
            length = _step_length(start_point, neighbor, pixel_size)
            visited_edges.add(ek)
            branch_points = [start_point, neighbor]
            prev = start_point
            current = neighbor
            while node_key(current) is None:
                next_points = [pt for pt in neighbor_map[current] if pt != prev]
                if not next_points:
                    break
                nxt = next_points[0]
                ek = edge_key(current, nxt)
                if ek in visited_edges:
                    break
                length += _step_length(current, nxt, pixel_size)
                visited_edges.add(ek)
                branch_points.append(nxt)
                prev, current = current, nxt
            end_node = node_key(current)
            branch_records.append(
                {
                    "start_node": start_node,
                    "end_node": end_node,
                    "length": float(length),
                    "points": branch_points,
                }
            )

    if not node_region_points:
        for start in points:
            for neighbor in neighbor_map[start]:
                ek = edge_key(start, neighbor)
                if ek in visited_edges:
                    continue
                length = _step_length(start, neighbor, pixel_size)
                visited_edges.add(ek)
                branch_points = [start, neighbor]
                prev = start
                current = neighbor
                while True:
                    next_points = [pt for pt in neighbor_map[current] if pt != prev]
                    if not next_points:
                        break
                    nxt = next_points[0]
                    ek = edge_key(current, nxt)
                    if ek in visited_edges:
                        break
                    length += _step_length(current, nxt, pixel_size)
                    visited_edges.add(ek)
                    branch_points.append(nxt)
                    prev, current = current, nxt
                    if current == start:
                        break
                branch_records.append(
                    {
                        "start_node": None,
                        "end_node": None,
                        "length": float(length),
                        "points": branch_points,
                    }
                )

    branch_records = _dedupe_branch_records(branch_records, node_key)
    endpoint_regions_by_id = _regions_by_id(endpoint_labels)
    junction_regions_by_id = _regions_by_id(junction_labels)
    return branch_records, endpoint_regions_by_id, junction_regions_by_id


def _skeleton_topology_details(
    mask: np.ndarray,
    pixel_size: tuple[float, ...],
    *,
    terminal_prune_length: float | None = None,
) -> dict[str, Any]:
    if np.asarray(mask).ndim == 3 and len(pixel_size) == 3:
        iso_mask, iso_skeleton, zoom_factors, iso_pixel_size = _skeletonize_object_isotropic(mask, pixel_size)
        if not np.any(iso_skeleton):
            empty = _empty_topology_details(np.asarray(mask, dtype=bool))
            fallback = _principal_axis_fallback_topology(np.asarray(mask, dtype=bool), pixel_size)
            return fallback if fallback is not None else empty
        iso_details = _topology_details_from_skeleton(
            iso_skeleton,
            iso_pixel_size,
            mask=iso_mask,
            terminal_prune_length=terminal_prune_length,
        )
        details = _project_isotropic_topology_details(iso_details, np.asarray(mask, dtype=bool), zoom_factors)
        if _needs_principal_axis_fallback(details):
            fallback = _principal_axis_fallback_topology(
                np.asarray(mask, dtype=bool),
                pixel_size,
            )
            if fallback is not None:
                return fallback
        return details

    skeleton = _skeletonize_lee(mask)
    if not np.any(skeleton):
        empty = _empty_topology_details(np.asarray(skeleton, dtype=bool))
        fallback = _principal_axis_fallback_topology(np.asarray(mask, dtype=bool), pixel_size)
        return fallback if fallback is not None else empty

    details = _topology_details_from_skeleton(
        skeleton,
        pixel_size,
        mask=mask,
        terminal_prune_length=terminal_prune_length,
    )
    if _needs_principal_axis_fallback(details):
        fallback = _principal_axis_fallback_topology(
            np.asarray(mask, dtype=bool),
            pixel_size,
        )
        if fallback is not None:
            return fallback
    return details


def _major_axis_length_in_units(prop, pixel_size: tuple[float, ...]) -> float:
    axis_major_length = getattr(prop, "axis_major_length", 0.0) or 0.0
    if axis_major_length <= 0:
        return 0.0
    spatial_scale = float(np.mean(pixel_size)) if pixel_size else 1.0
    return float(axis_major_length * spatial_scale)


def _perimeter_in_units(mask: np.ndarray, pixel_size: tuple[float, ...]) -> float:
    """Estimate 2D perimeter from the binary boundary and convert to physical units."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not np.any(binary):
        return 0.0
    spatial_scale = float(np.mean(tuple(float(v) for v in pixel_size[-2:]))) if pixel_size else 1.0
    return float(perimeter(binary, neighborhood=8) * spatial_scale)


def _surface_area_in_units(mask: np.ndarray, pixel_size: tuple[float, ...]) -> float:
    """Estimate 3D surface area from exposed voxel faces in physical units.

    This keeps the measurement aligned with the discrete voxel segmentation,
    rather than fitting a smoothed mesh as marching-cubes would.
    """
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3 or not np.any(binary):
        return 0.0
    vz, vy, vx = (float(v) for v in pixel_size[-3:])
    volume = binary.astype(np.uint8)

    left = np.pad(volume, ((0, 0), (0, 0), (1, 0)), mode="constant")[:, :, :-1]
    right = np.pad(volume, ((0, 0), (0, 0), (0, 1)), mode="constant")[:, :, 1:]
    front = np.pad(volume, ((0, 0), (1, 0), (0, 0)), mode="constant")[:, :-1, :]
    back = np.pad(volume, ((0, 0), (0, 1), (0, 0)), mode="constant")[:, 1:, :]
    up = np.pad(volume, ((1, 0), (0, 0), (0, 0)), mode="constant")[:-1, :, :]
    down = np.pad(volume, ((0, 1), (0, 0), (0, 0)), mode="constant")[1:, :, :]

    left_surface = np.clip(volume - left, 0, 1) * (vy * vz)
    right_surface = np.clip(volume - right, 0, 1) * (vy * vz)
    front_surface = np.clip(volume - front, 0, 1) * (vx * vz)
    back_surface = np.clip(volume - back, 0, 1) * (vx * vz)
    up_surface = np.clip(volume - up, 0, 1) * (vx * vy)
    down_surface = np.clip(volume - down, 0, 1) * (vx * vy)

    surface = (
        left_surface
        + right_surface
        + front_surface
        + back_surface
        + up_surface
        + down_surface
    )
    return float(np.sum(surface, dtype=np.float64))


def _circle_polygon(center: tuple[float, float], radius: float, num_vertices: int = 24) -> np.ndarray:
    cy, cx = center
    angles = np.linspace(0.0, 2.0 * np.pi, num_vertices, endpoint=False)
    ys = cy + radius * np.sin(angles)
    xs = cx + radius * np.cos(angles)
    return np.column_stack([ys, xs])


def _triangle_polygon(center: tuple[float, float], radius: float) -> np.ndarray:
    cy, cx = center
    angles = np.deg2rad(np.array([-90.0, 150.0, 30.0]))
    ys = cy + radius * np.sin(angles)
    xs = cx + radius * np.cos(angles)
    return np.column_stack([ys, xs])


class SegmentAnalysisMixin:
    def _analysis_current_frame_index(self, layer) -> int | None:
        dims_tag = _layer_dims_tag(layer)
        if dims_tag not in {"TYX", "TZYX"}:
            return None
        current_step = tuple(int(v) for v in getattr(self.viewer.dims, "current_step", ()))
        if not current_step:
            return 0
        try:
            layer_ndim = int(getattr(layer, "ndim", layer.data.ndim))
            viewer_ndim = int(getattr(self.viewer.dims, "ndim", len(current_step)))
            time_axis = max(viewer_ndim - layer_ndim, 0)
            return max(0, int(current_step[time_axis]))
        except (AttributeError, IndexError, TypeError, ValueError):
            return 0

    def _analysis_frame_key(self, layer) -> tuple[int, int | None]:
        return (id(layer), self._analysis_current_frame_index(layer))

    def _build_analysis_page(self) -> QWidget:
        ctrl_box = QGroupBox("Morphology Analysis")
        ctrl_grid = QGridLayout()
        ctrl_grid.setHorizontalSpacing(self._scaled_px(10))
        ctrl_grid.setVerticalSpacing(self._scaled_px(6))

        self._analysis_layer_combo = QComboBox()
        self._analysis_min_size_spin = QSpinBox()
        self._analysis_min_size_spin.setRange(0, 10**9)
        self._analysis_min_size_spin.setValue(20)
        self._analysis_min_size_spin.valueChanged.connect(self._on_analysis_min_size_changed)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_analysis_layers)
        analyze_objects_btn = QPushButton("Analyze Objects")
        analyze_objects_btn.clicked.connect(self._on_analyze_objects_clicked)
        refresh_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        analyze_objects_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layer_row = QHBoxLayout()
        layer_row.setContentsMargins(0, 0, 0, 0)
        layer_row.setSpacing(self._scaled_px(8))
        layer_label = QLabel("Layer")
        layer_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._analysis_layer_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layer_row.addWidget(layer_label)
        layer_row.addWidget(self._analysis_layer_combo, 1)
        ctrl_grid.addLayout(layer_row, 0, 0, 1, 2)
        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.setSpacing(self._scaled_px(8))
        filter_label = QLabel("Min pixel/voxel size")
        filter_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._analysis_min_size_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        filter_row.addWidget(filter_label)
        filter_row.addWidget(self._analysis_min_size_spin, 1)
        ctrl_grid.addLayout(filter_row, 1, 0, 1, 2)
        ctrl_grid.addWidget(refresh_btn, 2, 0)
        ctrl_grid.addWidget(analyze_objects_btn, 2, 1)
        self._analysis_frame_label = QLabel("Scope: static layer")
        with suppress(AttributeError):
            self._analysis_frame_label.setStyleSheet("color: #aab2bf;")
        ctrl_grid.addWidget(self._analysis_frame_label, 3, 0, 1, 2)
        ctrl_grid.setColumnStretch(0, 1)
        ctrl_grid.setColumnStretch(1, 1)
        ctrl_box.setLayout(ctrl_grid)
        with suppress(AttributeError):
            self._style_group_box(ctrl_box)

        self._analysis_summary = None

        plot_box = QGroupBox("Distribution Plots")
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.setContentsMargins(2, 4, 2, 4)
        if Figure is not None and FigureCanvasQTAgg is not None:
            bg, _fg, _grid, _bar = self._analysis_plot_palette()
            self._analysis_plot_figure = Figure(figsize=(4.6, 8.2), facecolor=bg)
            self._analysis_plot_axes = self._analysis_plot_figure.subplots(3, 1, gridspec_kw={"height_ratios": [1, 1, 1.8]})
            self._analysis_plot_canvas = FigureCanvasQTAgg(self._analysis_plot_figure)
            with suppress(AttributeError):
                self._analysis_plot_canvas.setMinimumHeight(self._scaled_px(640))
            self._apply_analysis_plot_theme()
            plot_scroll = QScrollArea()
            plot_scroll.setWidgetResizable(True)
            plot_scroll.setWidget(self._analysis_plot_canvas)
            self._analysis_plot_scroll = plot_scroll
            self._analysis_plot_canvas.installEventFilter(self)
            plot_scroll.viewport().installEventFilter(self)
            with suppress(AttributeError):
                plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            with suppress(AttributeError):
                plot_scroll.setFixedHeight(self._scaled_px(300))
            plot_layout.addWidget(plot_scroll)
            self._plot_distributions(None)
        else:
            plot_fallback = QTextEdit()
            plot_fallback.setReadOnly(True)
            plot_fallback.setPlainText("Matplotlib is unavailable in this environment, so the distribution plots cannot be shown.")
            self._analysis_plot_canvas = plot_fallback
            plot_layout.addWidget(plot_fallback)
        with suppress(AttributeError):
            plot_box.setMinimumHeight(self._scaled_px(320))
        with suppress(AttributeError):
            self._style_group_box(plot_box)

        table_box = QGroupBox("Measurements")
        table_layout = QVBoxLayout(table_box)
        export_row = QHBoxLayout()
        self._analysis_export_btn = QPushButton("Export")
        self._analysis_export_btn.clicked.connect(self._export_analysis_table)
        export_row.addWidget(self._analysis_export_btn)
        table_layout.addLayout(export_row)
        self._analysis_export_progress = QProgressBar()
        self._analysis_export_progress.setRange(0, 1)
        self._analysis_export_progress.setValue(0)
        self._analysis_export_progress.setFormat("")
        self._analysis_export_progress.setTextVisible(True)
        with suppress(AttributeError):
            self._analysis_export_progress.setMaximumHeight(self._scaled_px(18))
        self._analysis_export_progress.hide()
        table_layout.addWidget(self._analysis_export_progress)
        self._analysis_table = QTableWidget()
        self._analysis_table.setColumnCount(0)
        self._analysis_table.setRowCount(0)
        self._analysis_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._analysis_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._analysis_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._analysis_table.itemSelectionChanged.connect(self._on_analysis_table_selection_changed)
        with suppress(AttributeError):
            self._style_table_widget(self._analysis_table, min_height=180, max_height=280)
        table_layout.addWidget(self._analysis_table)
        with suppress(AttributeError):
            self._style_group_box(table_box)

        branch_box = QGroupBox("Branch Length List")
        branch_layout = QVBoxLayout(branch_box)
        branch_export_row = QHBoxLayout()
        self._analysis_branch_export_btn = QPushButton("Export")
        self._analysis_branch_export_btn.clicked.connect(self._export_analysis_branch_table)
        branch_export_row.addWidget(self._analysis_branch_export_btn)
        branch_layout.addLayout(branch_export_row)
        self._analysis_branch_table = QTableWidget()
        self._analysis_branch_table.setColumnCount(0)
        self._analysis_branch_table.setRowCount(0)
        self._analysis_branch_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._analysis_branch_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._analysis_branch_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._analysis_branch_table.itemSelectionChanged.connect(self._on_analysis_branch_table_selection_changed)
        with suppress(AttributeError):
            self._style_table_widget(self._analysis_branch_table, min_height=200, max_height=320)
        branch_layout.addWidget(self._analysis_branch_table)
        with suppress(AttributeError):
            self._style_group_box(branch_box)

        return self._build_panel_page(ctrl_box, plot_box, table_box, branch_box)

    def _rebuild_analysis_layer_combo(self, *, clear_outputs: bool) -> None:
        if self._analysis_layer_combo is None:
            return
        current_name = self._analysis_layer_combo.currentText()
        if clear_outputs:
            self._clear_analysis_results(reset_outputs=True)
        self._analysis_layer_combo.blockSignals(True)
        self._analysis_layer_combo.clear()
        for layer_item in self.viewer.layers:
            if _is_analysis_candidate_layer(layer_item):
                self._analysis_layer_combo.addItem(layer_item.name, layer_item)
        if self._analysis_layer_combo.count() > 0:
            idx = self._analysis_layer_combo.findText(current_name)
            self._analysis_layer_combo.setCurrentIndex(idx if idx >= 0 else 0)
        else:
            self._analysis_rows = []
        self._analysis_layer_combo.blockSignals(False)
        self._update_analysis_scope_label()

    def _refresh_analysis_layers(self) -> None:
        self._rebuild_analysis_layer_combo(clear_outputs=True)

    def _analysis_min_object_size(self) -> int:
        spin = getattr(self, "_analysis_min_size_spin", None)
        if spin is None:
            return 0
        with suppress(Exception):
            return max(0, int(spin.value()))
        return 0

    def _filtered_analysis_rows(self, rows: list[dict[str, float | int]] | None = None) -> list[dict[str, float | int]]:
        source_rows = self._analysis_all_rows if rows is None else rows
        return _filter_analysis_rows_by_min_size(source_rows, self._analysis_min_object_size())

    def _on_analysis_min_size_changed(self) -> None:
        filtered_rows = self._filtered_analysis_rows()
        self._populate_analysis_table(filtered_rows)
        self._plot_distributions(
            _filter_plot_info_for_rows(
                self._analysis_all_plot_info,
                filtered_rows,
                self._analysis_topology_cache,
            )
        )

    def _selected_analysis_layer(self):
        if self._analysis_layer_combo is None or self._analysis_layer_combo.count() == 0:
            return None
        return self._analysis_layer_combo.currentData()

    def _update_analysis_scope_label(self) -> None:
        if getattr(self, "_analysis_frame_label", None) is None:
            return
        layer = self._selected_analysis_layer()
        if layer is None:
            self._analysis_frame_label.setText("Scope: no layer selected")
            return
        frame_index = self._analysis_current_frame_index(layer)
        if frame_index is None:
            self._analysis_frame_label.setText("Scope: static layer")
        else:
            self._analysis_frame_label.setText(f"Scope: frame {int(frame_index)}")

    def _analysis_plot_palette(self) -> tuple[str, str, str, str]:
        return ("#262930", "#E6E6E6", "#6b7280", "#7fb3ff")

    def _apply_analysis_plot_theme(self) -> None:
        if (
            self._analysis_plot_canvas is None
            or self._analysis_plot_figure is None
            or self._analysis_plot_axes is None
        ):
            return
        bg, fg, grid, _bar = self._analysis_plot_palette()
        self._analysis_plot_canvas.setStyleSheet(f"background-color: {bg}; border: none;")
        self._analysis_plot_figure.set_facecolor(bg)
        self._analysis_plot_figure.patch.set_facecolor(bg)
        for ax in np.atleast_1d(self._analysis_plot_axes):
            ax.set_facecolor(bg)
            _style_ax(ax, fg, grid)

    def _analyze_segmentation_layer(self, layer):
        frame_index = self._analysis_current_frame_index(layer)
        binary = _component_mask_from_layer(layer, frame_index=frame_index)
        unit = _get_units_from_layer(layer)
        voxel_size = _get_pixel_size_tuple(layer, binary.ndim)
        labels, object_rows, plot_info, topology_cache = analyze_binary_components(
            binary,
            unit=unit,
            voxel_size=voxel_size,
            frame_index=frame_index,
        )
        self._analysis_topology_cache = topology_cache
        return labels, object_rows, plot_info

    def _apply_analysis_result(
        self,
        layer,
        labels: np.ndarray,
        rows: list[dict[str, float | int]],
        plot_info: dict[str, Any] | None,
        *,
        topology_cache: dict[int, dict[str, object]] | None = None,
    ) -> None:
        self._analysis_labels = labels.astype(np.int32)
        if topology_cache is not None:
            self._analysis_topology_cache = topology_cache
        md = dict(getattr(layer, "metadata", {}) or {})
        md["is_analysis_labels"] = True
        md["analysis_source_layer"] = layer.name
        md["is_segmentation"] = False
        frame_index = self._analysis_current_frame_index(layer)
        if frame_index is not None:
            md["analysis_frame"] = int(frame_index)
        kwargs = {"metadata": md}
        scale = _spatial_scale_for_ndim(layer, labels.ndim)
        if scale is not None:
            kwargs["scale"] = scale
        units = _spatial_units_for_ndim(layer, labels.ndim)
        if units is not None:
            kwargs["units"] = units
        labels_layer = self.viewer.add_labels(labels.astype(np.int32), name=self._next_layer_name(f"{layer.name} objects"), **kwargs)
        self._analysis_labels_layer = labels_layer
        self._analysis_selected_ids.clear()
        self._hide_related_analysis_layers(layer)
        self._set_analysis_aux_layers_visible(labels_opacity=1.0)
        self._arrange_analysis_overlay_layers()
        with suppress(Exception):
            self.viewer.layers.selection.active = labels_layer
        self._analysis_all_rows = list(rows)
        self._analysis_all_plot_info = dict(plot_info or {}) if plot_info is not None else None
        filtered_rows = self._filtered_analysis_rows(rows)
        self._populate_analysis_table(filtered_rows)
        self._plot_distributions(_filter_plot_info_for_rows(self._analysis_all_plot_info, filtered_rows, self._analysis_topology_cache))
        self._analysis_frame_state = self._analysis_frame_key(layer)
        self._update_analysis_scope_label()

    def _refresh_analysis_for_current_frame_if_needed(self, *, force: bool = False) -> None:
        layer = self._selected_analysis_layer()
        if layer is None:
            self._update_analysis_scope_label()
            return
        self._update_analysis_scope_label()
        frame_key = self._analysis_frame_key(layer)
        if not force and frame_key == getattr(self, "_analysis_frame_state", None):
            return
        if not force and frame_key == getattr(self, "_analysis_pending_frame_key", None):
            return
        if not force and self._analysis_labels_layer is None and not self._analysis_rows:
            return
        try:
            self._start_analysis_refresh(layer, frame_key=frame_key)
        except (TypeError, ValueError, AttributeError, RuntimeError):
            return

    def _analysis_column_labels(self, rows: list[dict[str, float | int]]) -> list[str]:
        if not rows:
            return []
        layer = self._selected_analysis_layer()
        unit = _get_units_from_layer(layer) if layer is not None else "um"
        labels: list[str] = []
        for column in self._analysis_columns(rows[0]):
            if column == "label":
                labels.append("object label")
            elif column == "frame":
                labels.append("frame")
            elif column == "object_label":
                labels.append("object label")
            elif column == "branch_index":
                labels.append("branch")
            elif column == "branch_length":
                labels.append(f"branch length ({unit})")
            elif column == "pixels":
                labels.append("pixels (count)")
            elif column == "voxels":
                labels.append("voxels (count)")
            elif column == "perimeter":
                labels.append(f"perimeter ({unit})")
            elif column == "area":
                labels.append(f"area ({unit}^2)")
            elif column == "surface_area":
                labels.append(f"surface area ({unit}^2)")
            elif column == "volume":
                labels.append(f"volume ({unit}^3)")
            elif column == "total_branch_length":
                labels.append(f"total branch length ({unit})")
            elif column == "branch_number":
                labels.append("branch number")
            elif column == "junction_number":
                labels.append("junction number")
            elif column == "mean_branch_length":
                labels.append(f"mean branch length ({unit})")
            elif column == "endpoint_number":
                labels.append("endpoint number")
            elif column == "length source":
                labels.append("length source")
            else:
                labels.append(column)
        return labels

    def _analysis_columns(self, row: dict[str, float | int]) -> list[str]:
        preferred = [
            "label",
            "frame",
            "object_label",
            "branch_index",
            "branch_length",
            "pixels",
            "voxels",
            "perimeter",
            "area",
            "surface_area",
            "volume",
            "branch_number",
            "junction_number",
            "endpoint_number",
            "total_branch_length",
            "mean_branch_length",
            "length source",
        ]
        columns = [column for column in preferred if column in row and not str(column).startswith("_")]
        columns.extend(column for column in row if column not in columns and not str(column).startswith("_"))
        return columns

    def _analysis_export_headers_and_rows(self) -> tuple[list[str], list[list[str]]]:
        if not self._analysis_rows:
            return [], []
        columns = self._analysis_columns(self._analysis_rows[0])
        headers = self._analysis_column_labels(self._analysis_rows)
        rows: list[list[str]] = []
        for row in self._analysis_rows:
            rows.append([f"{row[column]:.6g}" if isinstance(row[column], float) else str(row[column]) for column in columns])
        return headers, rows

    def _analysis_branch_rows(self) -> list[dict[str, float | int]]:
        layer = self._selected_analysis_layer()
        frame_index = self._analysis_current_frame_index(layer) if layer is not None else None
        return _analysis_branch_rows_from_rows(self._analysis_rows, self._analysis_topology_cache, frame_index)

    def _analysis_branch_export_headers_and_rows(self) -> tuple[list[str], list[list[str]]]:
        rows = self._analysis_branch_rows()
        if not rows:
            return [], []
        columns = self._analysis_columns(rows[0])
        headers = self._analysis_column_labels(rows)
        rows_out: list[list[str]] = []
        for row in rows:
            rows_out.append([f"{row[column]:.6g}" if isinstance(row[column], float) else str(row[column]) for column in columns])
        return headers, rows_out

    def _export_analysis_table(self) -> None:
        if self._cancel_analysis_export_if_running():
            return
        layer = self._selected_analysis_layer()
        if layer is not None and _layer_dims_tag(layer) in {"TYX", "TZYX"}:
            self._export_all_analysis_frames()
            return
        headers, rows = self._analysis_export_headers_and_rows()
        if not headers or not rows:
            QMessageBox.information(self, "No data", "Run morphology analysis first, then export the table.")
            return
        self._export_analysis_rows(headers, rows, title="Export measurements", default_name="mitochondria_measurements.csv")

    def _export_analysis_branch_table(self) -> None:
        if self._cancel_analysis_export_if_running():
            return
        layer = self._selected_analysis_layer()
        if layer is not None and _layer_dims_tag(layer) in {"TYX", "TZYX"}:
            self._export_all_analysis_branches()
            return
        headers, rows = self._analysis_branch_export_headers_and_rows()
        if not headers or not rows:
            QMessageBox.information(self, "No data", "Run morphology analysis first, then export the branch table.")
            return
        self._export_analysis_rows(headers, rows, title="Export branch lengths", default_name="mitochondria_branch_lengths.csv")

    def _export_analysis_rows(self, headers: list[str], rows: list[list[str]], *, title: str, default_name: str) -> None:
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            title,
            default_name,
            "CSV (*.csv);;Text (*.txt);;Excel Workbook (*.xlsx)",
        )
        if not path:
            return
        suffix = os.path.splitext(path)[1].lower()
        if not suffix:
            if "xlsx" in selected_filter.lower():
                path += ".xlsx"
                suffix = ".xlsx"
            elif "txt" in selected_filter.lower():
                path += ".txt"
                suffix = ".txt"
            else:
                path += ".csv"
                suffix = ".csv"
        try:
            write_analysis_export_file(path, headers, rows)
        except RuntimeError as e:
            if suffix == ".xlsx" and Workbook is None:
                QMessageBox.warning(self, "XLSX unavailable", "openpyxl is not available in this environment. Please export as CSV/TXT or install openpyxl.")
                return
            QMessageBox.critical(self, "Export failed", f"Failed to export table: {e!r}")
            return
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, "Export failed", f"Failed to export table: {e!r}")
            return
        QMessageBox.information(self, "Export complete", f"Saved measurements to:\n{path}")

    def _cancel_analysis_export_if_running(self) -> bool:
        if self._analysis_export_thread is None or self._analysis_export_worker is None:
            return False
        self._request_worker_cancel("_analysis_export_thread", "_analysis_export_worker")
        self._set_analysis_export_running_ui(True)
        return True

    def _export_all_analysis_frames(self) -> None:
        layer = self._selected_analysis_layer()
        if layer is None:
            QMessageBox.information(self, "No layer", "Please choose a segmentation layer first.")
            return
        dims_tag = _layer_dims_tag(layer)
        if dims_tag not in {"TYX", "TZYX"}:
            QMessageBox.information(self, "Static layer", "The selected layer is not temporal.")
            return
        if int(getattr(layer.data, "ndim", 0)) < 3:
            QMessageBox.information(self, "No data", "The selected layer does not contain temporal frames.")
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export measurements",
            "mitochondria_measurements.csv",
            "CSV (*.csv);;Text (*.txt);;Excel Workbook (*.xlsx)",
        )
        if not path:
            return
        suffix = os.path.splitext(path)[1].lower()
        if not suffix:
            if "xlsx" in selected_filter.lower():
                path += ".xlsx"
            elif "txt" in selected_filter.lower():
                path += ".txt"
            else:
                path += ".csv"
        self._start_analysis_export_all_frames(layer, path, export_mode="objects")

    def _export_all_analysis_branches(self) -> None:
        layer = self._selected_analysis_layer()
        if layer is None:
            QMessageBox.information(self, "No layer", "Please choose a segmentation layer first.")
            return
        dims_tag = _layer_dims_tag(layer)
        if dims_tag not in {"TYX", "TZYX"} or int(getattr(layer.data, "ndim", 0)) < 3:
            QMessageBox.information(self, "Static layer", "The selected layer is not temporal.")
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export branch lengths",
            "mitochondria_branch_lengths.csv",
            "CSV (*.csv);;Text (*.txt);;Excel Workbook (*.xlsx)",
        )
        if not path:
            return
        suffix = os.path.splitext(path)[1].lower()
        if not suffix:
            if "xlsx" in selected_filter.lower():
                path += ".xlsx"
            elif "txt" in selected_filter.lower():
                path += ".txt"
            else:
                path += ".csv"
        self._start_analysis_export_all_frames(layer, path, export_mode="branches")

    def _populate_analysis_table(self, rows: list[dict[str, float | int]]) -> None:
        if self._analysis_table is None:
            return
        self._analysis_rows = rows
        self._analysis_table.blockSignals(True)
        if not rows:
            self._analysis_table.clear()
            self._analysis_table.setRowCount(0)
            self._analysis_table.setColumnCount(0)
            self._analysis_table.blockSignals(False)
            self._populate_selected_analysis_table(set())
            self._populate_analysis_branch_table([])
            return
        columns = self._analysis_columns(rows[0])
        self._analysis_table.setColumnCount(len(columns))
        self._analysis_table.setHorizontalHeaderLabels(self._analysis_column_labels(rows))
        self._analysis_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            label_id = int(row["label"])
            for c, column in enumerate(columns):
                item = _make_table_item(row[column])
                item.setData(Qt.UserRole, label_id)
                self._analysis_table.setItem(r, c, item)
        self._analysis_table.resizeColumnsToContents()
        self._analysis_table.blockSignals(False)
        self._populate_selected_analysis_table(self._analysis_selected_ids)
        self._populate_analysis_branch_table(self._analysis_branch_rows())

    def _populate_selected_analysis_table(self, label_ids: set[int]) -> None:
        if self._analysis_selected_table is None:
            return
        self._analysis_selected_table.blockSignals(True)
        rows = [row for row in self._analysis_rows if int(row.get("label", 0)) in label_ids]
        if not rows:
            self._analysis_selected_table.clear()
            self._analysis_selected_table.setRowCount(0)
            self._analysis_selected_table.setColumnCount(0)
            self._analysis_selected_table.blockSignals(False)
            return
        columns = self._analysis_columns(rows[0])
        self._analysis_selected_table.setColumnCount(len(columns))
        self._analysis_selected_table.setHorizontalHeaderLabels(self._analysis_column_labels(rows))
        self._analysis_selected_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            label_id = int(row["label"])
            for c, column in enumerate(columns):
                item = _make_table_item(row[column])
                item.setData(Qt.UserRole, label_id)
                self._analysis_selected_table.setItem(r, c, item)
        self._analysis_selected_table.resizeColumnsToContents()
        self._analysis_selected_table.blockSignals(False)

    def _populate_analysis_branch_table(self, rows: list[dict[str, float | int]]) -> None:
        if self._analysis_branch_table is None:
            return
        self._analysis_branch_table.blockSignals(True)
        if not rows:
            self._analysis_branch_table.clear()
            self._analysis_branch_table.setRowCount(0)
            self._analysis_branch_table.setColumnCount(0)
            self._analysis_branch_table.blockSignals(False)
            self._update_analysis_branch_highlight(self._selected_analysis_layer(), set())
            return
        columns = self._analysis_columns(rows[0])
        self._analysis_branch_table.setColumnCount(len(columns))
        self._analysis_branch_table.setHorizontalHeaderLabels(self._analysis_column_labels(rows))
        self._analysis_branch_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            object_label = int(row.get("object_label", 0))
            object_branch_index = int(row.get("_object_branch_index", row.get("branch_index", 0)))
            for c, column in enumerate(columns):
                item = _make_table_item(row[column])
                item.setData(Qt.UserRole, object_label)
                item.setData(Qt.UserRole + 1, object_branch_index)
                self._analysis_branch_table.setItem(r, c, item)
        self._analysis_branch_table.resizeColumnsToContents()
        self._analysis_branch_table.blockSignals(False)

    def _event_has_multi_modifier(self, event) -> bool:
        modifiers = getattr(event, "modifiers", ()) or ()
        return any(any(token in str(modifier).lower() for token in ("control", "ctrl", "meta", "command")) for modifier in modifiers)

    def _select_analysis_rows_for_labels(self, label_ids: set[int]) -> None:
        if self._analysis_table is None:
            return
        if not label_ids:
            self._analysis_table.blockSignals(True)
            self._analysis_table.clearSelection()
            self._analysis_table.blockSignals(False)
            return
        first_item = None
        self._analysis_table.blockSignals(True)
        self._analysis_table.clearSelection()
        selection_model = self._analysis_table.selectionModel()
        for row in range(self._analysis_table.rowCount()):
            item = self._analysis_table.item(row, 0)
            if item is None or int(item.data(Qt.UserRole)) not in label_ids:
                continue
            if selection_model is not None:
                index = self._analysis_table.model().index(row, 0)
                selection_model.select(index, QItemSelectionModel.Select | QItemSelectionModel.Rows)
            if first_item is None:
                first_item = item
                self._analysis_table.setCurrentCell(row, 0)
        if first_item is not None:
            self._analysis_table.scrollToItem(first_item, QAbstractItemView.PositionAtCenter)
        self._analysis_table.blockSignals(False)
        self._analysis_table.setFocus()

    def _on_analysis_viewer_double_click(self, _viewer, event) -> None:
        panel_tabs = getattr(self, "_panel_tabs", None)
        analysis_tab = getattr(self, "_analysis_tab", None)
        if (
            panel_tabs is not None
            and analysis_tab is not None
            and panel_tabs.currentWidget() is not analysis_tab
        ):
            return

        for layer in (getattr(self, "_analysis_labels_layer", None),):
            if layer is None or not bool(getattr(layer, "visible", False)):
                continue
            try:
                if layer not in self.viewer.layers:
                    continue
                label_id = layer.get_value(
                    event.position,
                    view_direction=event.view_direction,
                    dims_displayed=event.dims_displayed,
                    world=True,
                ) or 0
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
            if int(label_id) <= 0:
                continue
            with suppress(Exception):
                self.viewer.layers.selection.active = layer
            self._on_analysis_layer_double_click(layer, event)
            with suppress(Exception):
                event.handled = True
            return

    def _on_analysis_layer_double_click(self, layer, event) -> None:
        label_id = layer.get_value(event.position, view_direction=event.view_direction, dims_displayed=event.dims_displayed, world=True) or 0
        if int(label_id) <= 0:
            return
        source_layer = self._selected_analysis_layer()
        if source_layer is None:
            return
        label_id = int(label_id)
        if self._event_has_multi_modifier(event):
            if label_id in self._analysis_selected_ids:
                self._analysis_selected_ids.remove(label_id)
            else:
                self._analysis_selected_ids.add(label_id)
        else:
            self._analysis_selected_ids = {label_id}
        self._select_analysis_rows_for_labels(self._analysis_selected_ids)
        self._update_highlight_for_labels(source_layer, self._analysis_selected_ids)

    def _plot_branch_length_lorenz(self, ax, plot_info: dict[str, Any] | None, *, fg: str, grid: str, bar: str) -> None:
        ax.clear()
        values = np.asarray((plot_info or {}).get("branch_length_values", []), dtype=float)
        ax.set_facecolor("#262930")
        _style_ax(ax, fg, grid)
        ax.set_title("Branch length Lorenz")
        ax.set_xlabel("cumulative branch fraction")
        ax.set_ylabel("cumulative branch-length fraction")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color=grid, alpha=0.5, linewidth=1.0)
        lorenz = _lorenz_curve(values)
        gini = 0.0
        if lorenz is not None:
            x, cum = lorenz
            gini = float(max(0.0, min(1.0, 1.0 - 2.0 * _np_trapz(cum, x))))
            ax.plot(x, cum, color=bar, linewidth=2.0)
            ax.fill_between(x, cum, color=bar, alpha=0.22)
        ax.grid(color=grid, alpha=0.2)
        ax.text(0.04, 0.96, f"Gini = {gini:.3f}", transform=ax.transAxes,
                color=fg, fontsize=9, verticalalignment="top")

    def _plot_distributions(self, plot_info: dict[str, Any] | None) -> None:
        if self._analysis_plot_axes is None or self._analysis_plot_canvas is None:
            return
        axes = np.atleast_1d(self._analysis_plot_axes)
        bg, fg, grid, bar = self._analysis_plot_palette()
        self._apply_analysis_plot_theme()
        if self._analysis_plot_figure is not None:
            self._analysis_plot_figure.subplots_adjust(left=0.14, right=0.94, top=0.94, bottom=0.07, hspace=0.78)
        histogram_specs = [("size_values", "size_name", "size_unit"), ("branch_values", "branch_name", "branch_unit")]
        for ax, (values_key, name_key, unit_key) in zip(axes[:2], histogram_specs, strict=False):
            ax.clear()
            values = np.asarray((plot_info or {}).get(values_key, []), dtype=float)
            title = ((plot_info or {}).get(name_key) or values_key.replace("_values", "")).capitalize()
            unit = (plot_info or {}).get(unit_key, "")
            ax.set_xlabel("count")
            ax.set_ylabel(f"{title.lower()} ({unit})" if unit else title.lower())
            ax.set_title(f"{title} distribution")
            _style_ax(ax, fg, grid)
            if values.size > 0:
                bins = min(max(values.size, 5), 20)
                counts, edges = np.histogram(values, bins=bins)
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.barh(centers, counts, height=np.diff(edges) * 0.9, color=bar, edgecolor=fg, alpha=0.9)
            ax.grid(axis="x", color=grid, alpha=0.25)
        if len(axes) >= 3:
            self._plot_branch_length_lorenz(axes[2], plot_info, fg=fg, grid=grid, bar=bar)
        self._analysis_plot_canvas.draw_idle()

    def _set_analysis_aux_layers_visible(self, *, labels_opacity: float) -> None:
        with suppress(Exception):
            if self._analysis_labels_layer is not None:
                self._analysis_labels_layer.opacity = labels_opacity
                self._analysis_labels_layer.visible = True
        with suppress(Exception):
            if self._analysis_topology_skeleton_layer is not None:
                self._analysis_topology_skeleton_layer.visible = True
        with suppress(Exception):
            if self._analysis_topology_markers_layer is not None:
                self._analysis_topology_markers_layer.visible = True

    def _hide_related_analysis_layers(self, layer) -> None:
        group_id = _get_group_id(layer) if layer is not None else None
        for layer_item in list(self.viewer.layers):
            if not hasattr(layer_item, "data"):
                continue
            if _is_analysis_aux_layer(layer_item):
                with suppress(Exception):
                    layer_item.visible = True
                if layer_item is self._analysis_labels_layer:
                    with suppress(Exception):
                        layer_item.opacity = 0.22
                continue
            if _is_related_group_layer(layer_item, layer, group_id) and layer_item is not self._analysis_highlight_layer:
                with suppress(Exception):
                    layer_item.visible = False

    def _restore_related_analysis_layers(self, layer) -> None:
        group_id = _get_group_id(layer) if layer is not None else None
        for layer_item in list(self.viewer.layers):
            if not hasattr(layer_item, "data"):
                continue
            if _is_analysis_aux_layer(layer_item):
                with suppress(Exception):
                    layer_item.visible = True
                if layer_item is self._analysis_labels_layer:
                    with suppress(Exception):
                        layer_item.opacity = 1.0
                continue
            if _is_related_group_layer(layer_item, layer, group_id) and layer_item is not self._analysis_highlight_layer:
                with suppress(Exception):
                    layer_item.visible = True

    def _remove_analysis_branch_highlight_layer(self) -> None:
        layer_item = getattr(self, "_analysis_branch_highlight_layer", None)
        if layer_item is not None:
            md = getattr(layer_item, "metadata", {}) or {}
            if md.get("is_analysis_branch_highlight"):
                with suppress(Exception):
                    self.viewer.layers.remove(layer_item)
        self._analysis_branch_highlight_layer = None

    def _remove_analysis_topology_layers(self) -> None:
        for attr in ("_analysis_topology_skeleton_layer", "_analysis_topology_markers_layer"):
            layer_item = getattr(self, attr)
            if layer_item is not None:
                md = getattr(layer_item, "metadata", {}) or {}
                if md.get("is_analysis_topology"):
                    with suppress(Exception):
                        self.viewer.layers.remove(layer_item)
            setattr(self, attr, None)

    def _clear_analysis_results(self, *, reset_outputs: bool) -> None:
        self._analysis_cleanup_in_progress = True
        try:
            with suppress(Exception):
                self._request_worker_cancel("_analysis_thread", "_analysis_worker")
            self._analysis_pending_frame_key = None
            if self._analysis_labels_layer is not None:
                md = getattr(self._analysis_labels_layer, "metadata", {}) or {}
                if md.get("is_analysis_labels"):
                    with suppress(Exception):
                        self.viewer.layers.remove(self._analysis_labels_layer)
                self._analysis_labels_layer = None
            if self._analysis_highlight_layer is not None:
                md = getattr(self._analysis_highlight_layer, "metadata", {}) or {}
                if md.get("is_analysis_highlight"):
                    with suppress(Exception):
                        self.viewer.layers.remove(self._analysis_highlight_layer)
                self._analysis_highlight_layer = None
            self._remove_analysis_branch_highlight_layer()
            self._remove_analysis_topology_layers()
            self._analysis_selected_ids.clear()
            self._analysis_labels = None
            self._analysis_frame_state = None
            self._analysis_topology_cache = {}
            self._analysis_all_rows = []
            self._analysis_all_plot_info = None
            if reset_outputs:
                self._analysis_rows = []
                self._populate_analysis_table([])
                self._populate_selected_analysis_table(set())
                self._plot_distributions(None)
        finally:
            self._analysis_cleanup_in_progress = False

    def _arrange_analysis_overlay_layers(self) -> None:
        order = [
            self._analysis_labels_layer,
            self._analysis_topology_skeleton_layer,
            self._analysis_topology_markers_layer,
            self._analysis_branch_highlight_layer,
        ]
        existing = [layer for layer in order if layer is not None and layer in self.viewer.layers]
        if not existing:
            return
        insert_at = min(self.viewer.layers.index(layer) for layer in existing)
        for layer in existing:
            with suppress(Exception):
                current_idx = self.viewer.layers.index(layer)
                self.viewer.layers.move(current_idx, insert_at)
                insert_at += 1

    def _update_analysis_branch_highlight(self, layer, selected_branches: set[tuple[int, int]]) -> None:
        if layer is None or self._analysis_labels is None or not selected_branches:
            self._remove_analysis_branch_highlight_layer()
            return
        highlight = np.zeros_like(self._analysis_labels, dtype=np.int32)
        for object_label, branch_index in sorted(selected_branches):
            cached = self._analysis_topology_cache.get(int(object_label)) or {}
            branch_records = list(cached.get("branch_records", ()))
            if branch_index < 1 or branch_index > len(branch_records):
                continue
            record = branch_records[int(branch_index) - 1]
            slices = cached.get("slice")
            if slices is None:
                continue
            local_mask = np.zeros(tuple(sl.stop - sl.start for sl in slices), dtype=bool)
            for point in record.get("points", ()):
                try:
                    local_mask[tuple(int(v) for v in point)] = True
                except (TypeError, ValueError, IndexError):
                    continue
            if np.any(local_mask):
                highlight_view = highlight[slices]
                highlight_view[local_mask] = 1
        if not np.any(highlight):
            self._remove_analysis_branch_highlight_layer()
            return
        metadata = {
            "analysis_source_layer": layer.name,
            "is_analysis_branch_highlight": True,
        }
        kwargs = {"name": f"{layer.name} branch selection", "metadata": metadata, "opacity": 1.0}
        scale = _spatial_scale_for_ndim(layer, highlight.ndim)
        if scale is not None:
            kwargs["scale"] = scale
        units = _spatial_units_for_ndim(layer, highlight.ndim)
        if units is not None:
            kwargs["units"] = units
        if self._analysis_branch_highlight_layer is None:
            self._analysis_branch_highlight_layer = self.viewer.add_labels(highlight, **kwargs)
        else:
            self._analysis_branch_highlight_layer.data = highlight
            self._analysis_branch_highlight_layer.name = kwargs["name"]
            self._analysis_branch_highlight_layer.metadata = dict(metadata)
            if scale is not None:
                self._analysis_branch_highlight_layer.scale = scale
            if units is not None:
                self._analysis_branch_highlight_layer.units = units
        with suppress(Exception):
            self._analysis_branch_highlight_layer.visible = True
            self._analysis_branch_highlight_layer.opacity = 1.0
            self._analysis_branch_highlight_layer.blending = "additive"
            self._analysis_branch_highlight_layer.color = {
                None: np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
                0: np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
                1: np.array([1.0, 0.35, 0.1, 1.0], dtype=float),
            }
        self._arrange_analysis_overlay_layers()

    def _build_selected_topology_visuals(self) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
        if self._analysis_labels is None or not self._analysis_selected_ids:
            ndim = int(self._analysis_labels.ndim) if self._analysis_labels is not None else 2
            return (
                np.zeros((0,) * ndim, dtype=np.int32),
                np.empty((0, ndim), dtype=float),
                [],
                np.empty((0,), dtype=float),
                np.empty((0, 4), dtype=float),
            )
        skeleton_mask = np.zeros_like(self._analysis_labels, dtype=np.int32)
        marker_coords: list[tuple[float, ...]] = []
        marker_symbols: list[str] = []
        marker_sizes: list[float] = []
        marker_colors: list[np.ndarray] = []
        for label_id in sorted(self._analysis_selected_ids):
            cached = self._analysis_topology_cache.get(int(label_id))
            if not cached:
                continue
            slices = cached["slice"]
            local_skeleton = np.asarray(cached["skeleton_mask"], dtype=bool)
            skeleton_view = skeleton_mask[slices]
            skeleton_view[local_skeleton] = 1
            offsets = np.array([float(sl.start or 0) for sl in slices], dtype=float)
            for region in cached["endpoint_regions"]:
                marker_coords.append(tuple(np.asarray(region["center"], dtype=float) + offsets))
                marker_symbols.append("disc")
                marker_sizes.append(1.1)
                marker_colors.append(np.array([1.0, 0.0, 0.0, 1.0], dtype=float))
            for region in cached["junction_regions"]:
                marker_coords.append(tuple(np.asarray(region["center"], dtype=float) + offsets))
                marker_symbols.append("triangle_up")
                marker_sizes.append(1.4 + 0.4 * max(np.sqrt(float(region["size"])) - 1.0, 0.0))
                marker_colors.append(np.array([0.0, 191.0 / 255.0, 1.0, 1.0], dtype=float))
        ndim = skeleton_mask.ndim
        coords_array = np.asarray(marker_coords, dtype=float) if marker_coords else np.empty((0, ndim), dtype=float)
        sizes_array = np.asarray(marker_sizes, dtype=float) if marker_sizes else np.empty((0,), dtype=float)
        colors_array = np.asarray(marker_colors, dtype=float) if marker_colors else np.empty((0, 4), dtype=float)
        return skeleton_mask, coords_array, marker_symbols, sizes_array, colors_array

    def _update_analysis_topology_layers(self, layer) -> None:
        if layer is None or self._analysis_labels is None or not self._analysis_selected_ids:
            self._remove_analysis_topology_layers()
            return
        skeleton_mask, marker_coords, marker_symbols, marker_sizes, marker_colors = self._build_selected_topology_visuals()
        metadata = {
            "analysis_source_layer": layer.name,
            "is_analysis_topology": True,
            "selected_object_labels": sorted(self._analysis_selected_ids),
        }
        scale = _spatial_scale_for_ndim(layer, skeleton_mask.ndim)
        units = _spatial_units_for_ndim(layer, skeleton_mask.ndim)
        skeleton_name = f"{layer.name} topology skeleton"
        if self._analysis_topology_skeleton_layer is None:
            kwargs = {"name": skeleton_name, "metadata": dict(metadata), "opacity": 1.0}
            if scale is not None:
                kwargs["scale"] = scale
            if units is not None:
                kwargs["units"] = units
            self._analysis_topology_skeleton_layer = self.viewer.add_labels(skeleton_mask, **kwargs)
        else:
            self._analysis_topology_skeleton_layer.data = skeleton_mask
            self._analysis_topology_skeleton_layer.name = skeleton_name
            self._analysis_topology_skeleton_layer.metadata = dict(metadata)
            if scale is not None:
                self._analysis_topology_skeleton_layer.scale = scale
            if units is not None:
                self._analysis_topology_skeleton_layer.units = units
        with suppress(Exception):
            self._analysis_topology_skeleton_layer.visible = True
            self._analysis_topology_skeleton_layer.opacity = 1.0
            self._analysis_topology_skeleton_layer.blending = "additive"
            self._analysis_topology_skeleton_layer.color = {
                None: np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
                0: np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
                1: np.array([1.0, 0.0, 1.0, 1.0], dtype=float),
            }

        markers_name = f"{layer.name} topology markers"
        if self._analysis_topology_markers_layer is not None:
            with suppress(Exception):
                self.viewer.layers.remove(self._analysis_topology_markers_layer)
            self._analysis_topology_markers_layer = None
        if len(marker_coords) == 0:
            self._arrange_analysis_overlay_layers()
            return
        if skeleton_mask.ndim == 2:
            shape_data = []
            shape_types = []
            for coord, symbol, size in zip(marker_coords, marker_symbols, marker_sizes, strict=False):
                center = (float(coord[0]), float(coord[1]))
                shape_data.append(_triangle_polygon(center, float(size)) if symbol == "triangle_up" else _circle_polygon(center, float(size)))
                shape_types.append("polygon")
            marker_kwargs = {
                "name": markers_name,
                "metadata": dict(metadata),
                "shape_type": shape_types,
                "face_color": marker_colors,
                "edge_color": marker_colors,
                "edge_width": 0.0,
                "opacity": 1.0,
            }
            if scale is not None:
                marker_kwargs["scale"] = scale
            if units is not None:
                marker_kwargs["units"] = units
            self._analysis_topology_markers_layer = self.viewer.add_shapes(shape_data, **marker_kwargs)
        else:
            marker_kwargs = {
                "name": markers_name,
                "metadata": dict(metadata),
                "size": np.asarray(marker_sizes, dtype=float) * 2.0,
                "face_color": marker_colors,
                "symbol": marker_symbols,
                "opacity": 1.0,
                "blending": "translucent_no_depth",
            }
            if scale is not None:
                marker_kwargs["scale"] = scale
            if units is not None:
                marker_kwargs["units"] = units
            self._analysis_topology_markers_layer = self.viewer.add_points(marker_coords, **marker_kwargs)
        with suppress(Exception):
            self._analysis_topology_markers_layer.visible = True
            self._analysis_topology_markers_layer.opacity = 1.0
        self._arrange_analysis_overlay_layers()

    def _update_highlight_for_labels(self, layer, label_ids: set[int] | None) -> None:
        active_label_ids = {int(label_id) for label_id in (label_ids or set()) if int(label_id) > 0}
        self._analysis_selected_ids = set(active_label_ids)
        self._populate_selected_analysis_table(self._analysis_selected_ids)
        if self._analysis_highlight_layer is not None:
            with suppress(Exception):
                self.viewer.layers.remove(self._analysis_highlight_layer)
            self._analysis_highlight_layer = None
        if not active_label_ids:
            self._remove_analysis_topology_layers()
            self._restore_related_analysis_layers(layer)
            return
        if layer is None or self._analysis_labels is None:
            return
        self._hide_related_analysis_layers(layer)
        self._update_analysis_topology_layers(layer)
        self._set_analysis_aux_layers_visible(labels_opacity=0.22)
        with suppress(Exception):
            if self._analysis_labels_layer is not None:
                self.viewer.layers.selection.active = self._analysis_labels_layer

    def _on_analysis_table_selection_changed(self) -> None:
        layer = self._selected_analysis_layer()
        if layer is None or self._analysis_table is None:
            return
        if self._analysis_branch_table is not None:
            self._analysis_branch_table.blockSignals(True)
            self._analysis_branch_table.clearSelection()
            self._analysis_branch_table.blockSignals(False)
        self._update_analysis_branch_highlight(layer, set())
        items = self._analysis_table.selectedItems()
        if not items:
            self._analysis_selected_ids.clear()
            self._update_highlight_for_labels(layer, set())
            return
        self._analysis_selected_ids = {
            int(item.data(Qt.UserRole))
            for item in items
            if item.data(Qt.UserRole) is not None
        }
        self._update_highlight_for_labels(layer, self._analysis_selected_ids)

    def _on_analysis_branch_table_selection_changed(self) -> None:
        layer = self._selected_analysis_layer()
        table = self._analysis_branch_table
        if layer is None or table is None:
            return
        items = table.selectedItems()
        if not items:
            return
        selected_labels: set[int] = set()
        for item in items:
            object_label = item.data(Qt.UserRole)
            if object_label is None:
                continue
            selected_labels.add(int(object_label))
        if selected_labels:
            self._analysis_selected_ids = selected_labels
            self._select_analysis_rows_for_labels(selected_labels)
            self._update_highlight_for_labels(layer, selected_labels)

    def _on_analyze_objects_clicked(self) -> None:
        layer = self._selected_analysis_layer()
        if layer is None:
            QMessageBox.information(self, "No segmentation", "Please choose a segmentation result layer first.")
            return
        try:
            self._start_analysis_refresh(layer, frame_key=self._analysis_frame_key(layer))
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            QMessageBox.critical(self, "Analysis failed", f"Failed to analyze segmentation objects: {e!r}")
