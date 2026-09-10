from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import polygon

from ._layer_candidates import is_binary_mask_data
from ._geometry import exposed_face_measure
from ._metadata import (
    layer_dims_tag as _layer_dims_tag,
    squeeze_leading_singletons as _squeeze_leading_singletons,
)


class ProximityCancelledError(InterruptedError):
    """Raised at a cooperative boundary when proximity work is cancelled."""


@dataclass(frozen=True)
class ProximityRoiSpec:
    roi_id: int
    name: str
    frame: int
    plane_index: tuple[int, ...]
    mask: np.ndarray


def _proximity_roi_specs_from_shapes(
    shape_data,
    *,
    data_shape: tuple[int, ...],
    dims_tag: str,
    current_step: tuple[int, ...],
) -> list[ProximityRoiSpec]:
    shape = tuple(int(value) for value in data_shape)
    if len(shape) not in {2, 3, 4} or any(value <= 0 for value in shape):
        return []

    step = tuple(int(value) for value in current_step)
    if len(step) < len(shape):
        step = (0,) * (len(shape) - len(step)) + step
    else:
        step = step[-len(shape) :]
    prefix_ndim = len(shape) - 2
    default_prefix = [
        max(0, min(step[axis], shape[axis] - 1))
        for axis in range(prefix_ndim)
    ]

    specs: list[ProximityRoiSpec] = []
    for roi_idx, vertices in enumerate(list(shape_data or []), start=1):
        try:
            coords = np.asarray(vertices, dtype=float)
        except (TypeError, ValueError):
            continue
        if (
            coords.ndim != 2
            or coords.shape[0] < 3
            or coords.shape[1] < 2
            or not np.all(np.isfinite(coords))
        ):
            continue
        if coords.shape[1] > len(shape):
            coords = coords[:, -len(shape) :]

        spatial_vertices = coords[:, -2:]
        if np.unique(spatial_vertices, axis=0).shape[0] < 3:
            continue

        plane_prefix = list(default_prefix)
        available_prefix = coords[:, :-2]
        prefix_to_copy = min(prefix_ndim, int(available_prefix.shape[1]))
        for offset in range(1, prefix_to_copy + 1):
            axis = prefix_ndim - offset
            coordinate = int(round(float(np.mean(available_prefix[:, -offset]))))
            plane_prefix[axis] = max(0, min(coordinate, shape[axis] - 1))

        rr, cc = polygon(
            spatial_vertices[:, 0],
            spatial_vertices[:, 1],
            shape=shape[-2:],
        )
        if rr.size == 0 or cc.size == 0:
            continue
        normalized_dims = str(dims_tag or "").upper()
        is_temporal_volume = normalized_dims == "TZYX" or len(shape) == 4
        is_temporal_2d = normalized_dims == "TYX"
        is_static_volume = normalized_dims == "ZYX" or (
            len(shape) == 3 and not is_temporal_2d
        )

        roi_mask = np.zeros(shape, dtype=bool)
        if is_temporal_volume:
            roi_mask[(plane_prefix[0], slice(None), rr, cc)] = True
        elif is_static_volume:
            roi_mask[(slice(None), rr, cc)] = True
        else:
            roi_mask[tuple(plane_prefix) + (rr, cc)] = True
        if not np.any(roi_mask):
            continue

        frame = (
            int(plane_prefix[0])
            if plane_prefix and (is_temporal_2d or is_temporal_volume)
            else -1
        )
        specs.append(
            ProximityRoiSpec(
                roi_id=roi_idx,
                name=f"ROI {roi_idx}",
                frame=frame,
                plane_index=tuple(plane_prefix),
                mask=roi_mask,
            )
        )
    return specs


def _proximity_roi_label_mask(
    roi_specs: list[ProximityRoiSpec],
    *,
    data_shape: tuple[int, ...],
) -> np.ndarray:
    shape = tuple(int(value) for value in data_shape)
    if len(shape) not in {2, 3, 4} or any(value <= 0 for value in shape):
        raise ValueError(f"Unsupported proximity ROI export shape: {shape}")

    max_label = max((int(spec.roi_id) for spec in roi_specs), default=0)
    dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.uint32
    labels = np.zeros(shape, dtype=dtype)
    for spec in roi_specs:
        roi_id = int(spec.roi_id)
        if roi_id <= 0:
            raise ValueError("Proximity ROI identifiers must be positive integers.")
        mask = np.asarray(spec.mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError(
                f"Proximity ROI shape mismatch: expected {shape}, got {mask.shape}."
            )
        labels[mask] = roi_id
    return labels


@dataclass(frozen=True)
class ProximitySummaryRow:
    roi_id: int
    roi_name: str
    frame: int
    roi_size: int
    roi_size_physical: float
    source_size: int
    target_size: int
    overlap_size: int
    overlap_size_physical: float
    source_proximity_size: int
    target_proximity_size: int
    source_proximity_size_physical: float
    target_proximity_size_physical: float
    source_proximity_fraction: float
    target_proximity_fraction: float
    proximity_manders_m1: float
    proximity_manders_m2: float
    source_mean_distance: float
    source_median_distance: float
    target_mean_distance: float
    target_median_distance: float
    manders_m1: float
    manders_m2: float
    dice: float
    jaccard: float


@dataclass(frozen=True)
class ProximityComponentRow:
    roi_id: int
    frame: int
    component_id: int
    global_component_id: int
    size: int
    size_physical: float
    source_label: int = 0
    source_size: int = 0
    source_size_physical: float = 0.0


@dataclass(frozen=True)
class ProximityResult:
    source_raw_name: str
    target_raw_name: str
    source_seg_name: str
    target_seg_name: str
    source_mask: np.ndarray
    target_mask: np.ndarray
    overlap_mask: np.ndarray
    source_proximity_mask: np.ndarray
    target_proximity_mask: np.ndarray
    proximity_mask: np.ndarray
    source_object_labels: np.ndarray
    component_labels: np.ndarray
    summary_rows: list[ProximitySummaryRow]
    component_rows: list[ProximityComponentRow]
    unit: str
    distance_unit: str
    voxel_measure: float
    spatial_power: int
    distance_threshold: float
    surface_only: bool


def _normalized_layer_array(layer, *, allow_float: bool) -> np.ndarray:
    data = np.asarray(getattr(layer, "data", None))
    if data.size == 0:
        raise ValueError("Layer has no data.")
    dims_tag = _layer_dims_tag(layer)
    expected_ndim = {"YX": 2, "TYX": 3, "ZYX": 3, "TZYX": 4}.get(dims_tag)
    if expected_ndim is not None:
        data = _squeeze_leading_singletons(data, expected_ndim)
        if data.ndim != expected_ndim:
            raise ValueError(
                f"Layer metadata declares {dims_tag}, but data has shape {data.shape}."
            )
    elif data.ndim > 4:
        data = _squeeze_leading_singletons(data, 4)
    if data.ndim not in {2, 3, 4}:
        raise ValueError(f"Unsupported ndim for proximity analysis: {data.ndim}")
    if allow_float and data.dtype.kind not in "buif":
        raise ValueError("Raw image data must be numeric.")
    # Raw intensities need no full-volume float64 conversion; reductions below
    # choose their accumulator dtype explicitly.
    return data


def _binary_from_layer(layer) -> np.ndarray:
    return _normalized_layer_array(layer, allow_float=False) > 0


def _instance_labels_from_layer(layer, mask: np.ndarray, voxel_size: tuple[float, ...]) -> np.ndarray:
    arr = _normalized_layer_array(layer, allow_float=False)
    mask = np.asarray(mask, dtype=bool)
    with np.errstate(all="ignore"):
        unique = np.unique(arr)
    nonzero = unique[unique != 0]
    integer_dtype = arr.dtype.kind in "ui"
    if nonzero.size > 1 and (integer_dtype or np.allclose(arr, np.round(arr))):
        if integer_dtype:
            # Preserve uint64 IDs exactly; even np.round may pass through float.
            labels = np.array(arr, copy=True)
        else:
            if not np.isfinite(arr).all() or np.any(np.abs(arr) > 2**53):
                raise ValueError("Large instance IDs must use an integer array dtype.")
            labels = np.round(arr).astype(np.int64)
        labels[~mask] = 0
        return labels

    labels_out = np.zeros_like(mask, dtype=np.int32)
    next_label = 1
    if mask.ndim == 4:
        for frame in range(int(mask.shape[0])):
            labels, n = ndi.label(mask[frame])
            if n <= 0:
                continue
            valid = labels > 0
            labels_out[frame][valid] = labels[valid] + next_label - 1
            next_label += int(n)
        return labels_out
    if mask.ndim == 3 and len(voxel_size) == 2:  # TYX
        for frame in range(int(mask.shape[0])):
            labels, n = ndi.label(mask[frame])
            if n <= 0:
                continue
            valid = labels > 0
            labels_out[frame][valid] = labels[valid] + next_label - 1
            next_label += int(n)
        return labels_out

    labels, _n = ndi.label(mask)
    return np.asarray(labels, dtype=np.int32)


def _compact_object_ids(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index labels densely for distance maps, with an exact original-ID lookup."""
    original_ids = np.unique(labels)
    original_ids = original_ids[original_ids > 0]
    if original_ids.size > np.iinfo(np.int32).max:
        raise ValueError("Too many source objects for proximity analysis.")
    compact = np.zeros(labels.shape, dtype=np.int32)
    source = labels.reshape(-1)
    destination = compact.reshape(-1)
    # Bound the temporary int64 search indices for large volumes.
    for start in range(0, source.size, 1_048_576):
        block = source[start:start + 1_048_576]
        positive = block > 0
        destination[start:start + block.size][positive] = (
            np.searchsorted(original_ids, block[positive]).astype(np.int32) + 1
        )
    return compact, original_ids


def _raw_from_layer(layer) -> np.ndarray:
    return _normalized_layer_array(layer, allow_float=True)


def _voxel_measure_for_shape(shape: tuple[int, ...], voxel_size: tuple[float, ...]) -> float:
    if len(shape) == 2:
        return float(voxel_size[-2] * voxel_size[-1])
    if len(shape) == 3:
        if len(voxel_size) == 2:  # TYX
            return float(voxel_size[-2] * voxel_size[-1])
        return float(voxel_size[-3] * voxel_size[-2] * voxel_size[-1])
    if len(shape) == 4:  # TZYX
        return float(voxel_size[-3] * voxel_size[-2] * voxel_size[-1])
    raise ValueError(f"Unsupported proximity shape: {shape}")


def _surface_measure_for_shape(shape: tuple[int, ...], voxel_size: tuple[float, ...]) -> float:
    if len(shape) == 2:
        sampling = tuple(float(v) for v in voxel_size[-2:])
    elif len(shape) == 3:
        sampling = tuple(float(v) for v in voxel_size[-2:]) if len(voxel_size) == 2 else tuple(float(v) for v in voxel_size[-3:])
    elif len(shape) == 4:
        sampling = tuple(float(v) for v in voxel_size[-3:])
    else:
        raise ValueError(f"Unsupported proximity shape: {shape}")
    if len(sampling) <= 1:
        return 1.0
    # Surface masks are voxel-shell approximations; use the smallest spatial
    # spacings as one boundary element measure.
    measure = 1.0
    for spacing in sorted(sampling)[: len(sampling) - 1]:
        measure *= float(spacing)
    return float(measure)


def _spatial_power_for_shape(shape: tuple[int, ...], voxel_size: tuple[float, ...]) -> int:
    if len(shape) == 2:
        return 2
    if len(shape) == 3:
        return 2 if len(voxel_size) == 2 else 3
    if len(shape) == 4:
        return 3
    raise ValueError(f"Unsupported proximity shape: {shape}")


def _format_frame_for_mask(mask: np.ndarray, fallback_frame: int) -> int:
    if mask.ndim in {2, 3}:
        return int(fallback_frame)
    return int(fallback_frame)


def _compute_component_rows_for_roi(
    overlap_mask: np.ndarray,
    *,
    roi_id: int,
    fallback_frame: int,
    voxel_measure: float,
    next_global_id: int,
    temporal_3d: bool = False,
    physical_size_fn=None,
) -> tuple[np.ndarray, list[ProximityComponentRow], int]:
    rows: list[ProximityComponentRow] = []
    labels_out = np.zeros_like(overlap_mask, dtype=np.int32)

    if overlap_mask.ndim == 4 or (overlap_mask.ndim == 3 and temporal_3d):
        for frame in range(int(overlap_mask.shape[0])):
            labels, n = ndi.label(overlap_mask[frame])
            if n <= 0:
                continue
            counts = np.bincount(labels.ravel())
            for comp_id in range(1, int(n) + 1):
                global_id = next_global_id
                next_global_id += 1
                labels_out[frame][labels == comp_id] = global_id
                size = int(counts[comp_id])
                component_selection = np.zeros_like(overlap_mask, dtype=bool)
                component_selection[frame][labels == comp_id] = True
                size_physical = (
                    float(physical_size_fn(component_selection))
                    if physical_size_fn is not None
                    else float(size * voxel_measure)
                )
                rows.append(
                    ProximityComponentRow(
                        roi_id=roi_id,
                        frame=frame,
                        component_id=comp_id,
                        global_component_id=global_id,
                        size=size,
                        size_physical=size_physical,
                    )
                )
        return labels_out, rows, next_global_id

    labels, n = ndi.label(overlap_mask)
    if n > 0:
        counts = np.bincount(labels.ravel())
        frame_value = _format_frame_for_mask(overlap_mask, fallback_frame)
        for comp_id in range(1, int(n) + 1):
            global_id = next_global_id
            next_global_id += 1
            labels_out[labels == comp_id] = global_id
            size = int(counts[comp_id])
            component_selection = labels == comp_id
            size_physical = (
                float(physical_size_fn(component_selection))
                if physical_size_fn is not None
                else float(size * voxel_measure)
            )
            rows.append(
                ProximityComponentRow(
                    roi_id=roi_id,
                    frame=frame_value,
                    component_id=comp_id,
                    global_component_id=global_id,
                    size=size,
                    size_physical=size_physical,
                )
            )
    return labels_out, rows, next_global_id


def _compute_target_proximity_rows_for_source_objects(
    source_labels: np.ndarray,
    target_basis: np.ndarray,
    roi_mask: np.ndarray,
    *,
    roi_id: int,
    fallback_frame: int,
    voxel_measure: float,
    proximity_measure: float,
    distance_threshold: float,
    target_distance_to_source: np.ndarray,
    nearest_source_labels: np.ndarray,
    next_global_id: int,
    target_physical_size_fn=None,
    source_id_values: np.ndarray | None = None,
) -> tuple[np.ndarray, list[ProximityComponentRow], int]:
    rows: list[ProximityComponentRow] = []
    labels_out = np.zeros_like(source_labels, dtype=np.int32)
    source_labels = np.asarray(source_labels, dtype=np.int32)
    target_basis = np.logical_and(np.asarray(target_basis, dtype=bool), np.asarray(roi_mask, dtype=bool))
    target_proximity = np.logical_and(target_basis, np.asarray(target_distance_to_source, dtype=float) <= float(distance_threshold))

    object_ids = [int(v) for v in np.unique(source_labels[np.asarray(roi_mask, dtype=bool)]) if int(v) > 0]
    object_records: list[tuple[int, int]] = []
    for object_id in object_ids:
        source_object = np.logical_and(source_labels == int(object_id), roi_mask)
        source_size = int(source_object.sum())
        if source_size <= 0:
            continue
        object_records.append((source_size, object_id))

    object_records.sort(key=lambda item: (-int(item[0]), int(item[1])))
    for source_size, object_id in object_records:
        original_id = int(source_id_values[object_id - 1]) if source_id_values is not None else object_id
        source_object = np.logical_and(source_labels == object_id, roi_mask)
        target_near_object = np.logical_and(target_proximity, nearest_source_labels == object_id)
        global_id = next_global_id
        next_global_id += 1
        labels_out[target_near_object] = global_id
        target_near_size = int(target_near_object.sum())
        target_near_size_physical = (
            float(target_physical_size_fn(target_near_object))
            if target_physical_size_fn is not None
            else float(target_near_size * proximity_measure)
        )
        rows.append(
            ProximityComponentRow(
                roi_id=roi_id,
                frame=_format_frame_for_mask(source_object, fallback_frame),
                component_id=original_id,
                global_component_id=global_id,
                size=target_near_size,
                size_physical=target_near_size_physical,
                source_label=original_id,
                source_size=int(source_size),
                source_size_physical=float(source_size * voxel_measure),
            )
        )
    return labels_out, rows, next_global_id


def _is_proximity_aux_layer_metadata(md: dict) -> bool:
    return any(
        bool(md.get(key))
        for key in (
            "is_proximity_result",
            "is_proximity_overlap_display",
            "is_proximity_component_highlight",
            "is_proximity_source_object_highlight",
            "is_proximity_roi",
            "is_proximity_roi_label",
            "is_proximity_roi_mask",
        )
    )


def is_proximity_segmentation_candidate_layer(layer) -> bool:
    if layer is None or not hasattr(layer, "data"):
        return False
    layer_type = str(getattr(layer, "_type_string", "") or "").lower()
    layer_class = layer.__class__.__name__.lower()
    if layer_type not in {"image", "labels"} and layer_class not in {"image", "labels"}:
        return False
    md = getattr(layer, "metadata", {}) or {}
    if _is_proximity_aux_layer_metadata(md):
        return False
    if md.get("is_analysis_labels") or md.get("is_analysis_highlight") or md.get("is_analysis_topology"):
        return False
    data = getattr(layer, "data", None)
    if data is None or int(getattr(data, "size", 0)) == 0:
        return False
    shape = tuple(int(value) for value in getattr(data, "shape", ()))
    squeezed_shape = tuple(value for value in shape if value != 1)
    if len(squeezed_shape) not in {2, 3, 4}:
        return False
    if md.get("is_segmentation"):
        return True
    try:
        return is_binary_mask_data(data)
    except (TypeError, ValueError, AttributeError):
        return False


def is_proximity_raw_candidate_layer(layer) -> bool:
    if layer is None or not hasattr(layer, "data"):
        return False
    layer_type = str(getattr(layer, "_type_string", "") or "").lower()
    layer_class = layer.__class__.__name__.lower()
    if layer_type not in {"image", "labels"} and layer_class not in {"image", "labels"}:
        return False
    md = getattr(layer, "metadata", {}) or {}
    if _is_proximity_aux_layer_metadata(md):
        return False
    if md.get("is_analysis_labels") or md.get("is_analysis_highlight") or md.get("is_analysis_topology"):
        return False
    if md.get("is_segmentation") or md.get("is_tracking"):
        return False
    data = getattr(layer, "data", None)
    if data is None or int(getattr(data, "size", 0)) == 0:
        return False
    shape = tuple(int(value) for value in getattr(data, "shape", ()))
    squeezed_shape = tuple(value for value in shape if value != 1)
    return len(squeezed_shape) in {2, 3, 4}


def _manders(raw: np.ndarray, seg_mask: np.ndarray, overlap_mask: np.ndarray) -> float:
    seg_intensity = float(np.asarray(raw[seg_mask], dtype=float).sum())
    if seg_intensity <= 0.0:
        return 0.0
    overlap_intensity = float(np.asarray(raw[overlap_mask], dtype=float).sum())
    return overlap_intensity / seg_intensity


def _spatial_sampling_for_shape(shape: tuple[int, ...], voxel_size: tuple[float, ...]) -> tuple[float, ...]:
    if len(shape) == 2:
        return tuple(float(v) for v in voxel_size[-2:])
    if len(shape) == 3:
        if len(voxel_size) == 2:  # TYX
            return tuple(float(v) for v in voxel_size[-2:])
        return tuple(float(v) for v in voxel_size[-3:])
    if len(shape) == 4:  # TZYX
        return tuple(float(v) for v in voxel_size[-3:])
    raise ValueError(f"Unsupported proximity shape: {shape}")


def _for_each_spatial_frame(mask: np.ndarray, voxel_size: tuple[float, ...], fn) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim == 4:
        return np.stack([fn(arr[frame]) for frame in range(arr.shape[0])], axis=0)
    if arr.ndim == 3 and len(voxel_size) == 2:  # TYX
        return np.stack([fn(arr[frame]) for frame in range(arr.shape[0])], axis=0)
    return fn(arr)


def _distance_to_mask(mask: np.ndarray, voxel_size: tuple[float, ...]) -> np.ndarray:
    sampling = _spatial_sampling_for_shape(np.asarray(mask).shape, voxel_size)

    def _distance_one(frame_mask: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame_mask, dtype=bool)
        if not np.any(frame):
            return np.full(frame.shape, np.inf, dtype=np.float32)
        return ndi.distance_transform_edt(~frame, sampling=sampling)

    return np.asarray(_for_each_spatial_frame(np.asarray(mask, dtype=bool), voxel_size, _distance_one), dtype=np.float32)


def _nearest_labels_to_mask(labels: np.ndarray, voxel_size: tuple[float, ...]) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    sampling = _spatial_sampling_for_shape(labels.shape, voxel_size)

    def _nearest_one(frame_labels: np.ndarray) -> np.ndarray:
        frame_labels = np.asarray(frame_labels, dtype=np.int32)
        source_mask = frame_labels > 0
        if not np.any(source_mask):
            return np.zeros_like(frame_labels, dtype=np.int32)
        indices = ndi.distance_transform_edt(~source_mask, sampling=sampling, return_distances=False, return_indices=True)
        nearest = frame_labels[tuple(indices)]
        return np.asarray(nearest, dtype=np.int32)

    if labels.ndim == 4:
        return np.stack([_nearest_one(labels[frame]) for frame in range(labels.shape[0])], axis=0)
    if labels.ndim == 3 and len(voxel_size) == 2:  # TYX
        return np.stack([_nearest_one(labels[frame]) for frame in range(labels.shape[0])], axis=0)
    return _nearest_one(labels)


def _surface_mask(mask: np.ndarray, voxel_size: tuple[float, ...]) -> np.ndarray:
    def _surface_one(frame_mask: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame_mask, dtype=bool)
        if not np.any(frame):
            return np.zeros_like(frame, dtype=bool)
        structure = ndi.generate_binary_structure(frame.ndim, 1)
        eroded = ndi.binary_erosion(frame, structure=structure, border_value=0)
        return np.logical_and(frame, ~eroded)

    return np.asarray(_for_each_spatial_frame(np.asarray(mask, dtype=bool), voxel_size, _surface_one), dtype=bool)


def _surface_measure_for_selection(object_mask: np.ndarray, selection_mask: np.ndarray, voxel_size: tuple[float, ...]) -> float:
    object_mask = np.asarray(object_mask, dtype=bool)
    selection_mask = np.asarray(selection_mask, dtype=bool)
    if object_mask.shape != selection_mask.shape:
        raise ValueError("Surface selection must match object mask shape.")
    if object_mask.ndim == 4 or (object_mask.ndim == 3 and len(voxel_size) == 2):
        return float(sum(exposed_face_measure(frame, voxel_size, selection_mask[index])
                         for index, frame in enumerate(object_mask)))
    return exposed_face_measure(object_mask, voxel_size, selection_mask)


def _selection_physical_measure(
    object_mask: np.ndarray,
    selection_mask: np.ndarray,
    voxel_size: tuple[float, ...],
    *,
    surface_only: bool,
) -> float:
    selection_mask = np.asarray(selection_mask, dtype=bool)
    if surface_only:
        return _surface_measure_for_selection(object_mask, selection_mask, voxel_size)
    return float(selection_mask.sum() * _voxel_measure_for_shape(selection_mask.shape, voxel_size))


def _distance_stats(distance_map: np.ndarray, basis_mask: np.ndarray) -> tuple[float, float]:
    values = np.asarray(distance_map, dtype=float)[np.asarray(basis_mask, dtype=bool)]
    if values.size == 0:
        return 0.0, 0.0
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.median(values))


def compute_proximity_result(
    source_raw_layer,
    source_seg_layer,
    target_raw_layer,
    target_seg_layer,
    *,
    roi_specs: list[ProximityRoiSpec] | None = None,
    voxel_size: tuple[float, ...] = (1.0, 1.0),
    distance_voxel_size: tuple[float, ...] | None = None,
    unit: str = "um",
    distance_unit: str | None = None,
    distance_threshold: float = 0.2,
    surface_only: bool = True,
    cancel_check=None,
) -> ProximityResult:
    def check_cancelled():
        if cancel_check is not None and cancel_check():
            raise ProximityCancelledError("Proximity analysis cancelled.")

    check_cancelled()
    source_mask_all = _binary_from_layer(source_seg_layer)
    target_mask_all = _binary_from_layer(target_seg_layer)
    source_raw_all = _raw_from_layer(source_raw_layer)
    target_raw_all = _raw_from_layer(target_raw_layer)

    if source_mask_all.shape != target_mask_all.shape:
        raise ValueError(
            f"Source and target segmentations must have the same shape, got {source_mask_all.shape} and {target_mask_all.shape}."
        )
    if source_raw_all.shape != source_mask_all.shape:
        raise ValueError(
            f"Source raw and source segmentation must have the same shape, got {source_raw_all.shape} and {source_mask_all.shape}."
        )
    if target_raw_all.shape != target_mask_all.shape:
        raise ValueError(
            f"Target raw and target segmentation must have the same shape, got {target_raw_all.shape} and {target_mask_all.shape}."
        )

    full_image_mode = not roi_specs
    if not roi_specs:
        roi_specs = [
            ProximityRoiSpec(
                roi_id=1,
                name="Full image",
                frame=-1,
                plane_index=(),
                mask=np.ones_like(source_mask_all, dtype=bool),
            )
        ]

    voxel_measure = _voxel_measure_for_shape(source_mask_all.shape, voxel_size)
    proximity_measure = _surface_measure_for_shape(source_mask_all.shape, voxel_size) if surface_only else voxel_measure
    spatial_power = _spatial_power_for_shape(source_mask_all.shape, voxel_size)
    temporal_3d = source_mask_all.ndim == 3 and len(voxel_size) == 2
    distance_voxel_size = tuple(float(v) for v in (distance_voxel_size or voxel_size))
    distance_unit = str(distance_unit or unit)
    threshold = max(0.0, float(distance_threshold))
    source_distance_to_target = _distance_to_mask(target_mask_all, distance_voxel_size)
    check_cancelled()
    target_distance_to_source = _distance_to_mask(source_mask_all, distance_voxel_size)
    check_cancelled()
    source_basis_all = _surface_mask(source_mask_all, voxel_size) if surface_only else source_mask_all
    target_basis_all = _surface_mask(target_mask_all, voxel_size) if surface_only else target_mask_all
    source_object_labels = _instance_labels_from_layer(source_seg_layer, source_mask_all, voxel_size)
    source_labels_all, source_id_values = _compact_object_ids(source_object_labels)
    nearest_source_labels_all = _nearest_labels_to_mask(source_labels_all, distance_voxel_size)
    check_cancelled()
    combined_source_mask = np.zeros_like(source_mask_all, dtype=bool)
    combined_target_mask = np.zeros_like(target_mask_all, dtype=bool)
    combined_overlap_mask = np.zeros_like(source_mask_all, dtype=bool)
    combined_source_proximity_mask = np.zeros_like(source_mask_all, dtype=bool)
    combined_target_proximity_mask = np.zeros_like(target_mask_all, dtype=bool)
    combined_proximity_mask = np.zeros_like(source_mask_all, dtype=bool)
    combined_component_labels = np.zeros_like(source_mask_all, dtype=np.int32)
    summary_rows: list[ProximitySummaryRow] = []
    component_rows: list[ProximityComponentRow] = []
    next_global_id = 1

    for spec in roi_specs:
        check_cancelled()
        roi_mask = np.asarray(spec.mask, dtype=bool)
        if roi_mask.shape != source_mask_all.shape:
            raise ValueError(
                f"ROI mask must match segmentation shape, got {roi_mask.shape} and {source_mask_all.shape}."
            )
        source_mask = np.logical_and(source_mask_all, roi_mask)
        target_mask = np.logical_and(target_mask_all, roi_mask)
        overlap_mask = np.logical_and(source_mask, target_mask)
        union_mask = np.logical_or(source_mask, target_mask)
        source_basis = np.logical_and(source_basis_all, roi_mask)
        target_basis = np.logical_and(target_basis_all, roi_mask)
        source_proximity_mask = np.logical_and(source_basis, source_distance_to_target <= threshold)
        target_proximity_mask = np.logical_and(target_basis, target_distance_to_source <= threshold)
        proximity_mask = target_proximity_mask

        combined_source_mask |= source_mask
        combined_target_mask |= target_mask
        combined_overlap_mask |= overlap_mask
        combined_source_proximity_mask |= source_proximity_mask
        combined_target_proximity_mask |= target_proximity_mask
        combined_proximity_mask |= proximity_mask

        source_size = int(source_mask.sum())
        target_size = int(target_mask.sum())
        overlap_size = int(overlap_mask.sum())
        union_size = int(union_mask.sum())
        roi_size = int(roi_mask.sum())
        source_basis_size = int(source_basis.sum())
        target_basis_size = int(target_basis.sum())
        source_proximity_size = int(source_proximity_mask.sum())
        target_proximity_size = int(target_proximity_mask.sum())
        source_proximity_size_physical = _selection_physical_measure(
            source_mask_all,
            source_proximity_mask,
            voxel_size,
            surface_only=surface_only,
        )
        target_proximity_size_physical = _selection_physical_measure(
            target_mask_all,
            target_proximity_mask,
            voxel_size,
            surface_only=surface_only,
        )
        source_proximity_fraction = (
            float(source_proximity_size / source_basis_size) if source_basis_size > 0 else 0.0
        )
        target_proximity_fraction = (
            float(target_proximity_size / target_basis_size) if target_basis_size > 0 else 0.0
        )

        source_raw = np.where(roi_mask, source_raw_all, 0)
        target_raw = np.where(roi_mask, target_raw_all, 0)
        manders_m1 = _manders(source_raw, source_mask, overlap_mask)
        manders_m2 = _manders(target_raw, target_mask, overlap_mask)
        proximity_manders_m1 = _manders(source_raw, source_basis, source_proximity_mask)
        proximity_manders_m2 = _manders(target_raw, target_basis, target_proximity_mask)
        source_mean_distance, source_median_distance = _distance_stats(source_distance_to_target, source_basis)
        target_mean_distance, target_median_distance = _distance_stats(target_distance_to_source, target_basis)
        denom = source_size + target_size
        dice = float((2.0 * overlap_size) / denom) if denom > 0 else 0.0
        jaccard = float(overlap_size / union_size) if union_size > 0 else 0.0

        if full_image_mode:
            roi_component_labels, roi_component_rows, next_global_id = _compute_target_proximity_rows_for_source_objects(
                source_labels_all,
                target_basis,
                roi_mask,
                roi_id=int(spec.roi_id),
                fallback_frame=int(spec.frame),
                voxel_measure=voxel_measure,
                proximity_measure=proximity_measure,
                distance_threshold=threshold,
                target_distance_to_source=target_distance_to_source,
                nearest_source_labels=nearest_source_labels_all,
                next_global_id=next_global_id,
                source_id_values=source_id_values,
                target_physical_size_fn=lambda selection: _selection_physical_measure(
                    target_mask_all,
                    selection,
                    voxel_size,
                    surface_only=surface_only,
                ),
            )
        else:
            roi_component_labels, roi_component_rows, next_global_id = _compute_component_rows_for_roi(
                proximity_mask,
                roi_id=int(spec.roi_id),
                fallback_frame=int(spec.frame),
                voxel_measure=proximity_measure,
                next_global_id=next_global_id,
                temporal_3d=temporal_3d,
                physical_size_fn=lambda selection: _selection_physical_measure(
                    target_mask_all,
                    selection,
                    voxel_size,
                    surface_only=surface_only,
                ),
            )
        nonzero = roi_component_labels > 0
        combined_component_labels[nonzero] = roi_component_labels[nonzero]
        component_rows.extend(roi_component_rows)
        summary_rows.append(
            ProximitySummaryRow(
                roi_id=int(spec.roi_id),
                roi_name=str(spec.name),
                frame=int(spec.frame),
                roi_size=roi_size,
                roi_size_physical=float(roi_size * voxel_measure),
                source_size=source_size,
                target_size=target_size,
                overlap_size=overlap_size,
                overlap_size_physical=float(overlap_size * voxel_measure),
                source_proximity_size=source_proximity_size,
                target_proximity_size=target_proximity_size,
                source_proximity_size_physical=float(source_proximity_size_physical),
                target_proximity_size_physical=float(target_proximity_size_physical),
                source_proximity_fraction=source_proximity_fraction,
                target_proximity_fraction=target_proximity_fraction,
                proximity_manders_m1=proximity_manders_m1,
                proximity_manders_m2=proximity_manders_m2,
                source_mean_distance=source_mean_distance,
                source_median_distance=source_median_distance,
                target_mean_distance=target_mean_distance,
                target_median_distance=target_median_distance,
                manders_m1=manders_m1,
                manders_m2=manders_m2,
                dice=dice,
                jaccard=jaccard,
            )
        )

    component_rows.sort(
        key=lambda row: (
            -int(row.source_size if int(row.source_size) > 0 else row.size),
            int(row.roi_id),
            int(row.frame),
            int(row.component_id),
        )
    )
    summary_rows.sort(key=lambda row: int(row.roi_id))

    return ProximityResult(
        source_raw_name=str(getattr(source_raw_layer, "name", "source raw")),
        target_raw_name=str(getattr(target_raw_layer, "name", "target raw")),
        source_seg_name=str(getattr(source_seg_layer, "name", "source segmentation")),
        target_seg_name=str(getattr(target_seg_layer, "name", "target segmentation")),
        source_mask=combined_source_mask,
        target_mask=combined_target_mask,
        overlap_mask=combined_overlap_mask,
        source_proximity_mask=combined_source_proximity_mask,
        target_proximity_mask=combined_target_proximity_mask,
        proximity_mask=combined_proximity_mask,
        source_object_labels=source_object_labels,
        component_labels=combined_component_labels,
        summary_rows=summary_rows,
        component_rows=component_rows,
        unit=str(unit),
        distance_unit=str(distance_unit),
        voxel_measure=float(voxel_measure),
        spatial_power=int(spatial_power),
        distance_threshold=threshold,
        surface_only=bool(surface_only),
    )
