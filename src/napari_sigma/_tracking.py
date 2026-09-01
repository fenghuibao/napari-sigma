from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import os

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree
from skimage.measure import label, regionprops

_WEIGHTED_CVT_ITERATIONS = 8


class TrackingCancelledError(InterruptedError):
    pass


@dataclass(frozen=True)
class TrackingConfig:
    max_distance: float = 50.0
    max_neighbors: int = 6
    distance_weight: float = 0.2
    overlap_weight: float = 0.8
    point_support_weight: float = 1.0
    cost_cutoff: float = 1.5
    time_limit: float | None = None
    event_delta_threshold: float = 0.1
    sample_points: int = 3000
    point_support_saturation: int = 3
    coverage_capacity_points: int = 0
    coverage_cap_skip_threshold: float = 0.7
    min_link_size: int = 20
    point_match_method: str = "greedy"
    point_unmatched_cost: float = 0.0
    min_object_match_fraction: float = 0.0
    tracking_workers: int = 0


@dataclass(frozen=True)
class Detection:
    id: int
    frame: int
    local_label: int
    centroid: tuple[float, ...]
    area: int
    image: np.ndarray
    slice_tuple: tuple[slice, ...]
    coords: np.ndarray
    sampled_coords: np.ndarray
    intensity_centroid: tuple[float, ...] | None = None
    intensity_weight: float = 0.0


@dataclass(frozen=True)
class LinkCandidate:
    id: int
    src: int
    dst: int
    cost: float
    raw_distance_cost: float
    raw_overlap_cost: float
    raw_point_support_cost: float
    distance_term: float
    overlap_term: float
    point_support_term: float
    distance: float
    overlap: float
    area_ratio: float
    point_distance: float
    average_point_distance: float
    matched_points: float
    src_fraction: float
    dst_fraction: float


@dataclass(frozen=True)
class TrackingEvent:
    kind: str
    frame: int
    frame_from: int
    frame_to: int
    sources: tuple[int, ...]
    targets: tuple[int, ...]


@dataclass(frozen=True)
class TrackingResult:
    frame_labels: np.ndarray
    tracked_labels: np.ndarray
    detections: list[Detection]
    links: list[LinkCandidate]
    selected_link_ids: tuple[int, ...]
    lineage_ids: dict[int, int]
    events: list[TrackingEvent]
    objective_value: float
    success: bool
    message: str


@dataclass(frozen=True)
class MatchFrameSummary:
    frame_from: int
    frame_to: int
    src_points: int
    dst_points: int
    matched_pairs: int
    src_coverage: float
    dst_coverage: float


@dataclass(frozen=True)
class MatchFrameDetail:
    frame_from: int
    frame_to: int
    src_points: np.ndarray
    dst_points: np.ndarray
    src_owner: np.ndarray
    dst_owner: np.ndarray
    match_src_indices: np.ndarray
    match_dst_indices: np.ndarray


@dataclass(frozen=True)
class MatchSummary:
    frame_summaries: list[MatchFrameSummary]
    frame_details: list[MatchFrameDetail]
    detections: list[Detection]
    total_src_points: int
    total_dst_points: int
    total_matched_pairs: int
    overall_src_coverage: float
    overall_dst_coverage: float
    message: str


def _spatial_ndim(segmentation: np.ndarray) -> int:
    if segmentation.ndim not in {3, 4}:
        raise ValueError("Segmentation stack must be TYX or TZYX (time-first).")
    return segmentation.ndim - 1


def _normalize_spacing(
    spacing: tuple[float, ...] | None,
    spatial_ndim: int,
) -> tuple[float, ...]:
    if spacing is None:
        return (1.0,) * spatial_ndim
    spacing = tuple(float(v) for v in spacing)
    if len(spacing) != spatial_ndim:
        raise ValueError(
            f"Expected spacing with {spatial_ndim} spatial values, got {spacing}."
        )
    if any(not np.isfinite(value) or value <= 0.0 for value in spacing):
        raise ValueError("Spacing values must be finite and positive.")
    return spacing


def _label_frame(frame: np.ndarray) -> np.ndarray:
    mask = np.asarray(frame) > 0
    return label(mask, connectivity=1)


def _label_stack(segmentation: np.ndarray) -> np.ndarray:
    return np.stack(
        [_label_frame(segmentation[t]) for t in range(segmentation.shape[0])],
        axis=0,
    ).astype(np.int32)


def _snap_centers_to_unique_voxels(
    coords: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if coords.shape[0] == 0 or centers.shape[0] == 0:
        return np.empty((0, coords.shape[1] if coords.ndim == 2 else 0), dtype=float)
    k = min(int(centers.shape[0]), int(coords.shape[0]))
    tree = cKDTree(coords)
    query_k = min(max(8, k // 8), int(coords.shape[0]))
    _dist, nearest = tree.query(centers[:k], k=query_k)
    nearest = np.asarray(nearest)
    if nearest.ndim == 1:
        nearest = nearest[:, None]

    selected: list[int] = []
    used: set[int] = set()
    for row in nearest:
        for idx in np.ravel(row):
            point_idx = int(idx)
            if point_idx in used:
                continue
            selected.append(point_idx)
            used.add(point_idx)
            break

    if len(selected) < k:
        order = np.argsort(np.asarray(weights, dtype=float))[::-1]
        for idx in order:
            point_idx = int(idx)
            if point_idx in used:
                continue
            selected.append(point_idx)
            used.add(point_idx)
            if len(selected) >= k:
                break

    return coords[np.asarray(selected[:k], dtype=int)]


def _weighted_cvt_sample_points(
    coords: np.ndarray,
    intensities: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return coords
    sample_count = max(1, min(int(sample_count), int(coords.shape[0])))

    values = np.asarray(intensities, dtype=float).reshape(-1)
    if values.shape[0] != coords.shape[0]:
        raise ValueError(
            "Raw intensity samples must match the object voxel coordinates."
        )
    weights = np.where(
        np.isfinite(values) & (values > 0.0),
        values,
        0.0,
    )
    positive = weights[weights > 0.0]
    if positive.size == 0:
        raise ValueError(
            "Raw image has no positive finite intensity inside a segmented object."
        )
    # Small deterministic floor: dim regions keep representatives, while bright
    # regions still attract more CVT centers.
    weights = weights + 0.05 * float(np.median(positive))
    if coords.shape[0] <= sample_count:
        return coords

    chosen = np.empty(sample_count, dtype=int)
    first = int(np.argmax(weights))
    chosen[0] = first
    min_dist_sq = np.sum((coords - coords[first][None, :]) ** 2, axis=1)
    min_dist_sq[first] = -1.0
    for idx in range(1, sample_count):
        scores = min_dist_sq * weights
        scores[chosen[:idx]] = -1.0
        next_idx = int(np.argmax(scores))
        chosen[idx] = next_idx
        dist_sq = np.sum((coords - coords[next_idx][None, :]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)
        min_dist_sq[chosen[: idx + 1]] = -1.0
    centers = coords[chosen].copy()
    for _idx in range(_WEIGHTED_CVT_ITERATIONS):
        tree = cKDTree(centers)
        dists, labels = tree.query(coords, k=1)
        new_centers = centers.copy()
        for center_idx in range(sample_count):
            members = labels == center_idx
            if not np.any(members):
                farthest_idx = int(np.argmax(dists * weights))
                new_centers[center_idx] = coords[farthest_idx]
                labels[farthest_idx] = center_idx
                dists[farthest_idx] = 0.0
                continue
            member_coords = coords[members]
            member_weights = weights[members]
            weight_sum = float(np.sum(member_weights))
            new_centers[center_idx] = (
                np.sum(member_coords * member_weights[:, None], axis=0) / weight_sum
            )
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return _snap_centers_to_unique_voxels(coords, centers, weights)


def _intensity_weighted_centroid(
    coords: np.ndarray,
    intensities: np.ndarray | None,
) -> tuple[tuple[float, ...] | None, float]:
    if intensities is None:
        return None, 0.0
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(intensities, dtype=float).reshape(-1)
    if coords.ndim != 2 or coords.shape[0] == 0 or values.shape[0] != coords.shape[0]:
        return None, 0.0
    finite = np.isfinite(values)
    if not np.any(finite):
        return None, 0.0
    weights = np.where(finite, values, 0.0)
    weights = weights - float(np.min(weights[finite]))
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return None, 0.0
    centroid = np.sum(coords * weights[:, None], axis=0) / weight_sum
    return tuple(float(v) for v in centroid), weight_sum


def _allocate_frame_sample_counts(
    props: list,
    frame_intensity: np.ndarray | None,
    config: TrackingConfig,
) -> dict[int, int] | None:
    target = int(config.sample_points)
    if target <= 0:
        return None
    if frame_intensity is None:
        raise ValueError(
            "A raw intensity image is required when sample_points is greater than zero."
        )

    eligible: list[tuple[int, int, float]] = []
    for prop in props:
        area = int(prop.area)
        coords = np.asarray(prop.coords, dtype=np.intp)
        values = np.asarray(frame_intensity[tuple(coords.T)], dtype=float)
        positive_values = np.where(
            np.isfinite(values) & (values > 0.0),
            values,
            0.0,
        )
        integrated_intensity = float(
            np.sum(positive_values, dtype=np.float64)
        )
        eligible.append((int(prop.label), area, integrated_intensity))
    if not eligible:
        return {}

    weights = np.asarray([item[2] for item in eligible], dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError(
            "Raw image has no positive finite integrated object intensity for "
            "sample allocation."
        )

    capacities = np.asarray([item[1] for item in eligible], dtype=int)
    target = min(int(target), int(np.sum(capacities)))
    if target < len(eligible):
        raise ValueError(
            f"sample_points={target} is smaller than the {len(eligible)} objects "
            "in a frame; increase the budget so every object can be sampled."
        )

    counts = np.ones(len(eligible), dtype=int)
    remaining = int(target - len(eligible))
    if remaining > 0:
        exact = weights / total_weight * float(remaining)
        additions = np.minimum(
            np.floor(exact).astype(int),
            capacities - counts,
        )
        counts += additions
        remaining -= int(np.sum(additions))
        fractions = exact - np.floor(exact)
        order = np.lexsort((-weights, -fractions))
        while remaining > 0:
            added = 0
            for idx in order:
                idx = int(idx)
                if counts[idx] >= capacities[idx]:
                    continue
                counts[idx] += 1
                remaining -= 1
                added += 1
                if remaining <= 0:
                    break
            if added == 0:
                break

    return {
        int(label): int(count)
        for (label, _area, _weight), count in zip(eligible, counts, strict=False)
    }


def _extract_detections(
    frame_labels: np.ndarray,
    config: TrackingConfig,
    intensity_image: np.ndarray | None = None,
    cancel_check: callable | None = None,
) -> tuple[list[Detection], dict[int, list[int]]]:
    props_by_frame = [list(regionprops(frame_labels[t])) for t in range(frame_labels.shape[0])]
    detections: list[Detection] = []
    by_frame: dict[int, list[int]] = {}
    det_id = 0
    for t, props in enumerate(props_by_frame):
        if cancel_check is not None and cancel_check():
            raise TrackingCancelledError("Tracking cancelled.")
        frame_intensity = None
        if intensity_image is not None:
            candidate = np.asarray(intensity_image[t])
            if candidate.shape == frame_labels[t].shape:
                frame_intensity = candidate
        frame_sample_counts = _allocate_frame_sample_counts(
            props,
            frame_intensity,
            config,
        )
        for prop in props:
            full_coords = np.asarray(prop.coords, dtype=np.int32)
            if frame_sample_counts is None:
                sample_count = int(full_coords.shape[0])
            else:
                sample_count = int(frame_sample_counts.get(int(prop.label), 0))
            intensities = None
            if frame_intensity is not None:
                intensities = frame_intensity[tuple(full_coords.T)]
            intensity_centroid, intensity_weight = _intensity_weighted_centroid(full_coords, intensities)
            if frame_sample_counts is None:
                sampled_coords = full_coords.astype(float, copy=False)
            elif sample_count <= 0:
                sampled_coords = np.empty((0, full_coords.shape[1]), dtype=float)
            else:
                sampled_coords = _weighted_cvt_sample_points(
                    full_coords,
                    intensities,
                    sample_count,
                )
            detections.append(
                Detection(
                    id=det_id,
                    frame=t,
                    local_label=int(prop.label),
                    centroid=tuple(float(v) for v in prop.centroid),
                    area=int(prop.area),
                    image=np.asarray(prop.image, dtype=bool),
                    slice_tuple=tuple(prop.slice),
                    coords=full_coords,
                    sampled_coords=np.asarray(sampled_coords, dtype=float),
                    intensity_centroid=intensity_centroid,
                    intensity_weight=float(intensity_weight),
                )
            )
            by_frame.setdefault(t, []).append(det_id)
            det_id += 1
    return detections, by_frame


def _frame_point_arrays(
    detections: list[Detection],
    det_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    coords_list: list[np.ndarray] = []
    owner_list: list[np.ndarray] = []
    for det_id in det_ids:
        det = detections[det_id]
        pts = np.asarray(det.sampled_coords, dtype=float)
        if pts.size == 0:
            continue
        coords_list.append(pts)
        owner_list.append(np.full(pts.shape[0], det_id, dtype=np.int32))
    if not coords_list:
        spatial_ndim = detections[det_ids[0]].sampled_coords.shape[1] if det_ids else 2
        return np.zeros((0, spatial_ndim), dtype=float), np.zeros((0,), dtype=np.int32)
    return np.vstack(coords_list), np.concatenate(owner_list)


def _spacing_array(spacing: tuple[float, ...] | None, ndim: int) -> np.ndarray:
    if spacing is None:
        return np.ones((ndim,), dtype=float)
    values = np.asarray(spacing, dtype=float)
    if values.size != ndim:
        raise ValueError(f"Expected {ndim} spacing values, got {tuple(values.tolist())}.")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Spacing values must be finite and positive.")
    return values / float(values[-1])


def _scale_points(points: np.ndarray, spacing_scale: np.ndarray | None) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if spacing_scale is None:
        return points
    return points * np.asarray(spacing_scale, dtype=float).reshape(1, -1)


def _scale_point(point: np.ndarray, spacing_scale: np.ndarray | None) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    if spacing_scale is None:
        return point
    return point * np.asarray(spacing_scale, dtype=float)


def _shift_points(points: np.ndarray, shift: np.ndarray | None) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if shift is None or points.size == 0:
        return points
    return points - np.asarray(shift, dtype=float).reshape(1, -1)


def _frame_global_centroid(
    detections: list[Detection],
    det_ids: list[int],
    spatial_ndim: int,
) -> np.ndarray | None:
    weighted_sum = np.zeros((spatial_ndim,), dtype=float)
    total_weight = 0.0
    for det_id in det_ids:
        det = detections[det_id]
        if det.intensity_centroid is None or float(det.intensity_weight) <= 0.0:
            continue
        centroid = np.asarray(det.intensity_centroid, dtype=float)
        if centroid.shape[0] != spatial_ndim:
            continue
        weight = float(det.intensity_weight)
        weighted_sum += centroid * weight
        total_weight += weight
    if total_weight > 0.0:
        return weighted_sum / total_weight

    total = 0
    coord_sum = np.zeros((spatial_ndim,), dtype=float)
    for det_id in det_ids:
        coords = np.asarray(detections[det_id].coords, dtype=float)
        if coords.ndim != 2 or coords.shape[0] == 0:
            continue
        coord_sum += np.sum(coords, axis=0)
        total += int(coords.shape[0])
    if total <= 0:
        return None
    return coord_sum / float(total)


def _global_frame_shift(
    detections: list[Detection],
    src_det_ids: list[int],
    dst_det_ids: list[int],
    spatial_ndim: int,
) -> np.ndarray | None:
    src_centroid = _frame_global_centroid(detections, src_det_ids, spatial_ndim)
    dst_centroid = _frame_global_centroid(detections, dst_det_ids, spatial_ndim)
    if src_centroid is None or dst_centroid is None:
        return None
    return dst_centroid - src_centroid


def _detection_match_centroid(det: Detection) -> np.ndarray:
    if det.intensity_centroid is not None and float(det.intensity_weight) > 0.0:
        return np.asarray(det.intensity_centroid, dtype=float)
    return np.asarray(det.centroid, dtype=float)


def _candidate_point_edges(
    src_points: np.ndarray,
    src_owner: np.ndarray,
    dst_points: np.ndarray,
    detections_by_id: dict[int, Detection],
    config: TrackingConfig,
    spacing_scale: np.ndarray | None = None,
    dst_global_shift: np.ndarray | None = None,
) -> list[tuple[float, int, int, float]]:
    if src_points.size == 0 or dst_points.size == 0:
        return []

    src_metric_points = _scale_points(src_points, spacing_scale)
    aligned_dst_points = _shift_points(dst_points, dst_global_shift)
    dst_metric_points = _scale_points(aligned_dst_points, spacing_scale)
    tree_dst = cKDTree(dst_metric_points)
    tree_src = cKDTree(src_metric_points)
    edges: dict[tuple[int, int], tuple[float, float]] = {}
    radius = float(config.max_distance)
    max_neighbors = int(config.max_neighbors)
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be at least 1.")
    if np.isnan(radius) or radius < 0.0:
        raise ValueError("max_distance must be non-negative or infinity.")

    def _add_edge(src_idx: int, dst_idx: int, dist: float) -> None:
        if not np.isfinite(dist):
            return
        src_det = detections_by_id[int(src_owner[src_idx])]
        src_center = _scale_point(_detection_match_centroid(src_det), spacing_scale)
        src_radius = float(np.linalg.norm(src_metric_points[src_idx] - src_center))
        dst_radius = float(np.linalg.norm(dst_metric_points[dst_idx] - src_center))
        source_radius_diff = abs(src_radius - dst_radius)
        cost = float(dist) + float(source_radius_diff)
        key = (int(src_idx), int(dst_idx))
        prev = edges.get(key)
        if prev is None or cost < prev[0]:
            edges[key] = (cost, float(dist))

    dst_k = min(max_neighbors, int(dst_metric_points.shape[0]))
    for src_idx, point in enumerate(src_metric_points):
        dists, neighbor_ids = tree_dst.query(
            point,
            k=dst_k,
            distance_upper_bound=radius,
        )
        for dist, dst_idx in zip(
            np.atleast_1d(dists),
            np.atleast_1d(neighbor_ids),
            strict=False,
        ):
            if int(dst_idx) < int(dst_metric_points.shape[0]):
                _add_edge(int(src_idx), int(dst_idx), float(dist))

    src_k = min(max_neighbors, int(src_metric_points.shape[0]))
    for dst_idx, point in enumerate(dst_metric_points):
        dists, neighbor_ids = tree_src.query(
            point,
            k=src_k,
            distance_upper_bound=radius,
        )
        for dist, src_idx in zip(
            np.atleast_1d(dists),
            np.atleast_1d(neighbor_ids),
            strict=False,
        ):
            if int(src_idx) < int(src_metric_points.shape[0]):
                _add_edge(int(src_idx), int(dst_idx), float(dist))

    return sorted(
        [(cost, src_idx, dst_idx, dist) for (src_idx, dst_idx), (cost, dist) in edges.items()],
        key=lambda item: (item[0], item[3]),
    )


def _supplemental_coverage_assignment(
    edges: list[tuple[float, int, int, float]],
    src_owner: np.ndarray,
    dst_owner: np.ndarray,
    *,
    src_deficits: dict[int, int],
    dst_deficits: dict[int, int],
    exact_time_limit: float | None = None,
) -> list[int] | None:
    if not edges:
        return None

    src_owner = np.asarray(src_owner, dtype=np.int32)
    dst_owner = np.asarray(dst_owner, dtype=np.int32)
    edge_src = np.asarray([edge[1] for edge in edges], dtype=np.int32)
    edge_dst = np.asarray([edge[2] for edge in edges], dtype=np.int32)
    edge_cost = np.asarray([edge[0] for edge in edges], dtype=float)
    n_edges = len(edges)
    n_src = int(src_owner.size)
    n_dst = int(dst_owner.size)

    src_ids = sorted(int(det_id) for det_id in src_deficits)
    dst_ids = sorted(int(det_id) for det_id in dst_deficits)
    src_row = {det_id: idx for idx, det_id in enumerate(src_ids)}
    dst_row = {det_id: idx for idx, det_id in enumerate(dst_ids)}
    n_src_objects = len(src_ids)
    n_dst_objects = len(dst_ids)
    n_rows = n_src + n_dst + n_src_objects + n_dst_objects
    rows: list[int] = []
    cols: list[int] = []
    for col, (src_idx, dst_idx) in enumerate(zip(edge_src, edge_dst, strict=False)):
        rows.extend([int(src_idx), n_src + int(dst_idx)])
        cols.extend([col, col])
        src_id = int(src_owner[int(src_idx)])
        dst_id = int(dst_owner[int(dst_idx)])
        if src_id in src_row:
            rows.append(n_src + n_dst + src_row[src_id])
            cols.append(col)
        if dst_id in dst_row:
            rows.append(n_src + n_dst + n_src_objects + dst_row[dst_id])
            cols.append(col)
    matrix = coo_matrix(
        (np.ones(len(rows), dtype=float), (rows, cols)),
        shape=(n_rows, n_edges),
    ).tocsr()
    lower = np.concatenate(
        [
            np.zeros(n_src + n_dst, dtype=float),
            np.asarray([src_deficits[det_id] for det_id in src_ids], dtype=float),
            np.asarray([dst_deficits[det_id] for det_id in dst_ids], dtype=float),
        ]
    )
    upper = np.concatenate(
        [
            np.ones(n_src + n_dst, dtype=float),
            np.full(n_src_objects + n_dst_objects, np.inf, dtype=float),
        ]
    )
    constraint = LinearConstraint(matrix, lower, upper)
    bounds = Bounds(0.0, 1.0)

    # All variables are supplemental matches. Positive costs make the solver
    # add only the minimum matching needed to fill deficient objects.
    objective = edge_cost + 1e-9

    result = milp(
        objective,
        integrality=None,
        bounds=bounds,
        constraints=constraint,
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        if int(result.status) == 2:
            return None
        raise RuntimeError(f"Point-coverage optimization failed: {result.message}")

    values = np.asarray(result.x, dtype=float)
    fractional = (values > 1e-7) & (values < 1.0 - 1e-7)
    if np.any(fractional):
        options: dict[str, float | bool] = {"mip_rel_gap": 0.0, "presolve": True}
        if exact_time_limit is not None:
            options["time_limit"] = max(float(exact_time_limit), 1.0)
        result = milp(
            objective,
            integrality=np.ones(n_edges, dtype=np.int8),
            bounds=bounds,
            constraints=constraint,
            options=options,
        )
        if not result.success or result.x is None:
            raise RuntimeError(
                "Exact point-coverage optimization failed after a fractional LP "
                f"solution: {result.message}"
            )
        values = np.asarray(result.x, dtype=float)

    selected = np.flatnonzero(values > 0.5)
    src_added: dict[int, int] = defaultdict(int)
    dst_added: dict[int, int] = defaultdict(int)
    for idx in selected:
        src_added[int(src_owner[int(edge_src[idx])])] += 1
        dst_added[int(dst_owner[int(edge_dst[idx])])] += 1
    if any(src_added[det_id] < value for det_id, value in src_deficits.items()) or any(
        dst_added[det_id] < value for det_id, value in dst_deficits.items()
    ):
        raise RuntimeError("Supplemental point matching returned an invalid solution.")
    return [int(idx) for idx in selected]


def _supplement_unmatched_object_coverage(
    base_matches: list[tuple[int, int, float, float]],
    src_points: np.ndarray,
    src_owner: np.ndarray,
    dst_points: np.ndarray,
    dst_owner: np.ndarray,
    detections_by_id: dict[int, Detection],
    config: TrackingConfig,
    *,
    spacing_scale: np.ndarray | None = None,
    dst_global_shift: np.ndarray | None = None,
) -> list[tuple[int, int, float, float]]:
    fraction = float(getattr(config, "min_object_match_fraction", 0.0))
    if fraction <= 0.0:
        return base_matches

    src_owner = np.asarray(src_owner, dtype=np.int32)
    dst_owner = np.asarray(dst_owner, dtype=np.int32)
    matched_src_indices = {int(match[0]) for match in base_matches}
    matched_dst_indices = {int(match[1]) for match in base_matches}
    src_matched: dict[int, int] = defaultdict(int)
    dst_matched: dict[int, int] = defaultdict(int)
    for src_idx, dst_idx, _cost, _dist in base_matches:
        src_matched[int(src_owner[int(src_idx)])] += 1
        dst_matched[int(dst_owner[int(dst_idx)])] += 1

    src_ids, src_counts = np.unique(src_owner, return_counts=True)
    dst_ids, dst_counts = np.unique(dst_owner, return_counts=True)
    min_area = int(config.min_link_size)
    src_deficits = {
        int(det_id): int(np.ceil(fraction * int(count) - 1e-12))
        - src_matched[int(det_id)]
        for det_id, count in zip(src_ids, src_counts, strict=False)
        if detections_by_id[int(det_id)].area > min_area
        and src_matched[int(det_id)]
        < int(np.ceil(fraction * int(count) - 1e-12))
    }
    dst_deficits = {
        int(det_id): int(np.ceil(fraction * int(count) - 1e-12))
        - dst_matched[int(det_id)]
        for det_id, count in zip(dst_ids, dst_counts, strict=False)
        if detections_by_id[int(det_id)].area > min_area
        and dst_matched[int(det_id)]
        < int(np.ceil(fraction * int(count) - 1e-12))
    }
    if not src_deficits and not dst_deficits:
        return base_matches

    unmatched_src = np.asarray(
        [idx for idx in range(src_owner.size) if idx not in matched_src_indices],
        dtype=np.int32,
    )
    unmatched_dst = np.asarray(
        [idx for idx in range(dst_owner.size) if idx not in matched_dst_indices],
        dtype=np.int32,
    )
    if (
        int(sum(src_deficits.values())) > int(unmatched_dst.size)
        or int(sum(dst_deficits.values())) > int(unmatched_src.size)
    ):
        raise ValueError(
            "min_object_match_fraction is infeasible after freezing the initial links."
        )

    # The initial matches are frozen. Supplemental matching has no distance
    # cutoff because it only sees still-unmatched points and only retains edges
    # incident to an object below its coverage requirement.
    edges = _candidate_point_edges(
        np.asarray(src_points)[unmatched_src],
        src_owner[unmatched_src],
        np.asarray(dst_points)[unmatched_dst],
        detections_by_id,
        replace(config, max_distance=float("inf")),
        spacing_scale=spacing_scale,
        dst_global_shift=dst_global_shift,
    )
    edges = [
        edge
        for edge in edges
        if int(src_owner[unmatched_src[int(edge[1])]]) in src_deficits
        or int(dst_owner[unmatched_dst[int(edge[2])]]) in dst_deficits
    ]
    selected = _supplemental_coverage_assignment(
        edges,
        src_owner[unmatched_src],
        dst_owner[unmatched_dst],
        src_deficits=src_deficits,
        dst_deficits=dst_deficits,
        exact_time_limit=getattr(config, "time_limit", None),
    )
    if selected is None:
        raise ValueError(
            "Could not supplement unmatched points to satisfy "
            f"min_object_match_fraction={fraction:g}."
        )
    supplemental = [
        (
            int(unmatched_src[int(edges[idx][1])]),
            int(unmatched_dst[int(edges[idx][2])]),
            float(edges[idx][0]),
            float(edges[idx][3]),
        )
        for idx in selected
    ]
    return sorted(base_matches + supplemental, key=lambda item: (item[0], item[1]))


def _greedy_global_point_matching(
    src_points: np.ndarray,
    src_owner: np.ndarray,
    dst_points: np.ndarray,
    detections_by_id: dict[int, Detection],
    config: TrackingConfig,
    spacing_scale: np.ndarray | None = None,
    dst_global_shift: np.ndarray | None = None,
) -> list[tuple[int, int, float, float]]:
    edges = _candidate_point_edges(
        src_points,
        src_owner,
        dst_points,
        detections_by_id,
        config,
        spacing_scale=spacing_scale,
        dst_global_shift=dst_global_shift,
    )
    if not edges:
        return []

    matched_src: set[int] = set()
    matched_dst: set[int] = set()
    matches: list[tuple[int, int, float, float]] = []
    for cost, src_idx, dst_idx, dist in edges:
        if src_idx in matched_src or dst_idx in matched_dst:
            continue
        matched_src.add(src_idx)
        matched_dst.add(dst_idx)
        matches.append((src_idx, dst_idx, cost, dist))
    return matches


def _sparse_min_cost_assignment(
    n_rows: int,
    n_cols: int,
    edge_costs: dict[tuple[int, int], float],
    unmatched_cost: float = 0.0,
) -> list[tuple[int, int]]:
    """Return a sparse assignment with one private dummy per source row.

    Each real row receives a private dummy column, so rows without a valid
    partner can remain unmatched without materializing a dense cost matrix. A
    positive ``unmatched_cost`` enables partial matching; zero preserves the
    legacy maximum-cardinality behavior.
    """
    if n_rows <= 0 or n_cols <= 0 or not edge_costs:
        return []

    finite_costs = [float(v) for v in edge_costs.values() if np.isfinite(v)]
    if not finite_costs:
        return []
    max_real_cost = max(max(finite_costs), 0.0)
    if float(unmatched_cost) > 0.0:
        dummy_cost = float(unmatched_cost)
    else:
        dummy_cost = (max_real_cost + 1.0) * float(n_rows + 1)

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    epsilon = np.finfo(float).eps
    for (row, col), cost in edge_costs.items():
        if not np.isfinite(cost):
            continue
        rows.append(int(row))
        cols.append(int(col))
        values.append(max(float(cost), 0.0) + epsilon)
    for row in range(int(n_rows)):
        rows.append(row)
        cols.append(int(n_cols) + row)
        values.append(dummy_cost)

    matrix = coo_matrix(
        (np.asarray(values, dtype=float), (rows, cols)),
        shape=(int(n_rows), int(n_cols) + int(n_rows)),
    ).tocsr()
    row_ind, col_ind = min_weight_full_bipartite_matching(matrix)
    return [
        (int(row), int(col))
        for row, col in zip(row_ind, col_ind, strict=False)
        if int(col) < int(n_cols) and (int(row), int(col)) in edge_costs
    ]


def _matches_from_candidate_edges(
    edges: list[tuple[float, int, int, float]],
    n_src: int,
    n_dst: int,
    config: TrackingConfig,
) -> list[tuple[int, int, float, float]]:
    if not edges or n_src <= 0 or n_dst <= 0:
        return []

    edge_cost: dict[tuple[int, int], tuple[float, float]] = {}
    max_cost = 0.0
    for cost, src_idx, dst_idx, dist in edges:
        key = (int(src_idx), int(dst_idx))
        prev = edge_cost.get(key)
        if prev is None or float(cost) < prev[0]:
            edge_cost[key] = (float(cost), float(dist))
            max_cost = max(max_cost, float(cost))

    unmatched_cost = float(getattr(config, "point_unmatched_cost", 0.0))
    if n_src * n_dst > 8_000_000:
        assignments = _sparse_min_cost_assignment(
            n_src,
            n_dst,
            {key: value[0] for key, value in edge_cost.items()},
            unmatched_cost=unmatched_cost,
        )
        return [
            (
                int(src_idx),
                int(dst_idx),
                float(edge_cost[(src_idx, dst_idx)][0]),
                float(edge_cost[(src_idx, dst_idx)][1]),
            )
            for src_idx, dst_idx in assignments
        ]

    invalid_cost = max(
        max_cost + float(config.max_distance) * 10.0 + 1.0,
        unmatched_cost * float(n_src + 1) + 1.0,
        1.0e6,
    )
    dummy_columns = n_src if unmatched_cost > 0.0 else 0
    cost_matrix = np.full((n_src, n_dst + dummy_columns), invalid_cost, dtype=float)
    for (src_idx, dst_idx), (cost, _dist) in edge_cost.items():
        cost_matrix[src_idx, dst_idx] = cost
    if dummy_columns:
        rows = np.arange(n_src, dtype=int)
        cost_matrix[rows, n_dst + rows] = unmatched_cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches: list[tuple[int, int, float, float]] = []
    for src_idx, dst_idx in zip(row_ind, col_ind, strict=False):
        key = (int(src_idx), int(dst_idx))
        pair = edge_cost.get(key)
        if pair is None:
            continue
        cost, dist = pair
        matches.append((int(src_idx), int(dst_idx), float(cost), float(dist)))
    return matches


def _hungarian_global_point_matching(
    src_points: np.ndarray,
    src_owner: np.ndarray,
    dst_points: np.ndarray,
    dst_owner: np.ndarray,
    detections_by_id: dict[int, Detection],
    config: TrackingConfig,
    spacing_scale: np.ndarray | None = None,
    dst_global_shift: np.ndarray | None = None,
) -> list[tuple[int, int, float, float]]:
    n_src = int(np.asarray(src_points).shape[0])
    n_dst = int(np.asarray(dst_points).shape[0])
    if n_src <= 0 or n_dst <= 0:
        return []
    unmatched_cost = float(getattr(config, "point_unmatched_cost", 0.0))
    if not np.isfinite(unmatched_cost) or unmatched_cost < 0.0:
        raise ValueError("point_unmatched_cost must be finite and >= 0.")
    min_object_match_fraction = float(
        getattr(config, "min_object_match_fraction", 0.0)
    )
    if not np.isfinite(min_object_match_fraction) or not 0.0 <= min_object_match_fraction <= 1.0:
        raise ValueError("min_object_match_fraction must be finite and between 0 and 1.")

    edges = _candidate_point_edges(
        src_points,
        src_owner,
        dst_points,
        detections_by_id,
        config,
        spacing_scale=spacing_scale,
        dst_global_shift=dst_global_shift,
    )
    base_matches = _matches_from_candidate_edges(edges, n_src, n_dst, config)
    if min_object_match_fraction <= 0.0:
        return base_matches
    return _supplement_unmatched_object_coverage(
        base_matches,
        src_points,
        src_owner,
        dst_points,
        dst_owner,
        detections_by_id,
        config,
        spacing_scale=spacing_scale,
        dst_global_shift=dst_global_shift,
    )


def _global_point_matching(
    src_points: np.ndarray,
    src_owner: np.ndarray,
    dst_points: np.ndarray,
    dst_owner: np.ndarray,
    detections_by_id: dict[int, Detection],
    config: TrackingConfig,
    spacing_scale: np.ndarray | None = None,
    dst_global_shift: np.ndarray | None = None,
) -> list[tuple[int, int, float, float]]:
    method = str(getattr(config, "point_match_method", "greedy")).strip().lower()
    if method in {"", "greedy"}:
        return _greedy_global_point_matching(
            src_points,
            src_owner,
            dst_points,
            detections_by_id,
            config,
            spacing_scale=spacing_scale,
            dst_global_shift=dst_global_shift,
        )
    if method in {"hungarian", "linear_sum_assignment", "global"}:
        return _hungarian_global_point_matching(
            src_points,
            src_owner,
            dst_points,
            dst_owner,
            detections_by_id,
            config,
            spacing_scale=spacing_scale,
            dst_global_shift=dst_global_shift,
        )
    raise ValueError(
        "Unsupported point_match_method "
        f"{getattr(config, 'point_match_method', None)!r}. Use 'greedy' or 'hungarian'."
    )


def _links_from_point_matches(
    detections_by_id: dict[int, Detection],
    src_owner: np.ndarray,
    dst_owner: np.ndarray,
    matches: list[tuple[int, int, float, float]],
    config: TrackingConfig,
    start_id: int = 0,
) -> list[LinkCandidate]:
    if not matches:
        return []

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_dist_sums: dict[tuple[int, int], float] = defaultdict(float)
    pair_distances: dict[tuple[int, int], list[float]] = defaultdict(list)
    pair_min_point: dict[tuple[int, int], float] = {}

    for src_idx, dst_idx, _cost, dist in matches:
        src_id = int(src_owner[src_idx])
        dst_id = int(dst_owner[dst_idx])
        key = (src_id, dst_id)
        pair_counts[key] += 1
        pair_dist_sums[key] += float(dist)
        pair_distances[key].append(float(dist))
        prev = pair_min_point.get(key)
        if prev is None or float(dist) < prev:
            pair_min_point[key] = float(dist)

    match_distances = np.asarray([dist for _src_idx, _dst_idx, _cost, dist in matches], dtype=float)
    distance_norm = float(np.median(match_distances)) if match_distances.size > 0 else 1.0
    distance_norm = max(distance_norm, 1e-6)
    next_id = start_id
    links: list[LinkCandidate] = []
    for (src_id, dst_id), matched_count in sorted(pair_counts.items()):
        src = detections_by_id[src_id]
        dst = detections_by_id[dst_id]
        src_sample_count = len(src.sampled_coords)
        dst_sample_count = len(dst.sampled_coords)
        uncapped_src_fraction = min(
            float(matched_count) / float(max(src_sample_count, 1)),
            1.0,
        )
        uncapped_dst_fraction = min(
            float(matched_count) / float(max(dst_sample_count, 1)),
            1.0,
        )
        uncapped_coverage = 0.5 * (uncapped_src_fraction + uncapped_dst_fraction)
        skip_coverage_cap = (
            uncapped_coverage >= float(config.coverage_cap_skip_threshold)
        )
        coverage_sample_floor = 0
        if int(config.coverage_capacity_points) > 0 and not skip_coverage_cap:
            coverage_sample_floor = max(
                1,
                int(config.coverage_capacity_points),
            )
        src_coverage_denominator = max(src_sample_count, coverage_sample_floor, 1)
        dst_coverage_denominator = max(dst_sample_count, coverage_sample_floor, 1)
        matched_src_fraction = min(float(matched_count) / float(src_coverage_denominator), 1.0)
        matched_dst_fraction = min(float(matched_count) / float(dst_coverage_denominator), 1.0)
        overlap_like = 0.5 * (matched_src_fraction + matched_dst_fraction)
        average_point_distance = pair_dist_sums[(src_id, dst_id)] / max(matched_count, 1)
        median_point_distance = float(np.median(np.asarray(pair_distances[(src_id, dst_id)], dtype=float)))
        area_ratio = abs(src.area - dst.area) / max(src.area, dst.area, 1)
        point_support_saturation = max(1, int(config.point_support_saturation))
        point_support_capacity = max(
            min(
                len(src.sampled_coords),
                len(dst.sampled_coords),
                point_support_saturation,
            ),
            1,
        )
        point_support_fraction = min(
            float(matched_count) / float(point_support_capacity),
            1.0,
        )
        raw_point_support_cost = 1.0 - point_support_fraction
        raw_distance_cost = float(median_point_distance) / distance_norm
        raw_overlap_cost = 1.0 - overlap_like
        point_support_term = (
            float(config.point_support_weight) * raw_point_support_cost
        )
        distance_term = float(config.distance_weight) * raw_distance_cost
        overlap_term = float(config.overlap_weight) * raw_overlap_cost
        cost = distance_term + overlap_term + point_support_term
        links.append(
            LinkCandidate(
                id=next_id,
                src=src_id,
                dst=dst_id,
                cost=float(cost),
                raw_distance_cost=float(raw_distance_cost),
                raw_overlap_cost=float(raw_overlap_cost),
                raw_point_support_cost=float(raw_point_support_cost),
                distance_term=float(distance_term),
                overlap_term=float(overlap_term),
                point_support_term=float(point_support_term),
                distance=float(median_point_distance),
                overlap=float(overlap_like),
                area_ratio=float(area_ratio),
                point_distance=float(pair_min_point[(src_id, dst_id)]),
                average_point_distance=float(average_point_distance),
                matched_points=float(matched_count),
                src_fraction=float(matched_src_fraction),
                dst_fraction=float(matched_dst_fraction),
            )
        )
        next_id += 1
    return links


def compute_match_summary(
    segmentation: np.ndarray,
    spacing: tuple[float, ...] | None = None,
    config: TrackingConfig | None = None,
    intensity_image: np.ndarray | None = None,
    cancel_check: callable | None = None,
) -> MatchSummary:
    config = config or TrackingConfig()
    segmentation = np.asarray(segmentation)
    spatial_ndim = _spatial_ndim(segmentation)
    spacing_scale = _spacing_array(_normalize_spacing(spacing, spatial_ndim), spatial_ndim)
    if intensity_image is not None:
        intensity_image = np.asarray(intensity_image)
        if intensity_image.shape != segmentation.shape:
            raise ValueError(
                "intensity_image must match segmentation shape, got "
                f"{intensity_image.shape} and {segmentation.shape}."
            )

    frame_labels = _label_stack(segmentation)
    detections, by_frame = _extract_detections(
        frame_labels,
        config,
        intensity_image=intensity_image,
        cancel_check=cancel_check,
    )
    if not detections:
        return MatchSummary(
            frame_summaries=[],
            frame_details=[],
            detections=[],
            total_src_points=0,
            total_dst_points=0,
            total_matched_pairs=0,
            overall_src_coverage=0.0,
            overall_dst_coverage=0.0,
            message="No detections.",
        )

    detections_by_id = {det.id: det for det in detections}
    max_frame = max(by_frame.keys(), default=-1)
    frame_summaries: list[MatchFrameSummary] = []
    frame_details: list[MatchFrameDetail] = []
    total_src_points = 0
    total_dst_points = 0
    total_matched_pairs = 0

    for t in range(max_frame):
        if cancel_check is not None and cancel_check():
            raise TrackingCancelledError("Match cancelled.")
        src_ids = [
            idx for idx in by_frame.get(t, []) if len(detections_by_id[idx].sampled_coords) > 0
        ]
        dst_ids = [
            idx for idx in by_frame.get(t + 1, []) if len(detections_by_id[idx].sampled_coords) > 0
        ]
        src_points, src_owner = _frame_point_arrays(detections, src_ids)
        dst_points, dst_owner = _frame_point_arrays(detections, dst_ids)
        dst_global_shift = _global_frame_shift(
            detections,
            by_frame.get(t, []),
            by_frame.get(t + 1, []),
            spatial_ndim,
        )
        matches = _global_point_matching(
            src_points,
            src_owner,
            dst_points,
            dst_owner,
            detections_by_id,
            config,
            spacing_scale=spacing_scale,
            dst_global_shift=dst_global_shift,
        )
        src_n = int(src_points.shape[0])
        dst_n = int(dst_points.shape[0])
        matched_n = len(matches)
        total_src_points += src_n
        total_dst_points += dst_n
        total_matched_pairs += matched_n
        frame_summaries.append(
            MatchFrameSummary(
                frame_from=t,
                frame_to=t + 1,
                src_points=src_n,
                dst_points=dst_n,
                matched_pairs=matched_n,
                src_coverage=(matched_n / src_n) if src_n > 0 else 0.0,
                dst_coverage=(matched_n / dst_n) if dst_n > 0 else 0.0,
            )
        )
        frame_details.append(
            MatchFrameDetail(
                frame_from=t,
                frame_to=t + 1,
                src_points=np.asarray(src_points, dtype=float),
                dst_points=np.asarray(dst_points, dtype=float),
                src_owner=np.asarray(src_owner, dtype=np.int32),
                dst_owner=np.asarray(dst_owner, dtype=np.int32),
                match_src_indices=np.asarray([src_idx for src_idx, _dst_idx, _cost, _dist in matches], dtype=np.int32),
                match_dst_indices=np.asarray([dst_idx for _src_idx, dst_idx, _cost, _dist in matches], dtype=np.int32),
            )
        )

    overall_src = (total_matched_pairs / total_src_points) if total_src_points > 0 else 0.0
    overall_dst = (total_matched_pairs / total_dst_points) if total_dst_points > 0 else 0.0
    return MatchSummary(
        frame_summaries=frame_summaries,
        frame_details=frame_details,
        detections=detections,
        total_src_points=total_src_points,
        total_dst_points=total_dst_points,
        total_matched_pairs=total_matched_pairs,
        overall_src_coverage=overall_src,
        overall_dst_coverage=overall_dst,
        message="Global interior point match summary completed. [tracking-global-point-v1]",
    )


def _build_link_candidates(
    detections: list[Detection],
    by_frame: dict[int, list[int]],
    config: TrackingConfig,
    spacing_scale: np.ndarray | None = None,
    cancel_check: callable | None = None,
) -> list[LinkCandidate]:
    max_frame = max(by_frame.keys(), default=-1)
    detections_by_id = {det.id: det for det in detections}
    spatial_ndim = len(detections[0].centroid) if detections else 2
    frames = list(range(max_frame))

    def build_for_frame(t: int) -> tuple[int, list[LinkCandidate]]:
        if cancel_check is not None and cancel_check():
            raise TrackingCancelledError("Tracking cancelled.")
        src_ids = [
            idx for idx in by_frame.get(t, []) if len(detections[idx].sampled_coords) > 0
        ]
        dst_ids = [
            idx for idx in by_frame.get(t + 1, []) if len(detections[idx].sampled_coords) > 0
        ]
        if not src_ids or not dst_ids:
            return t, []

        src_points, src_owner = _frame_point_arrays(detections, src_ids)
        dst_points, dst_owner = _frame_point_arrays(detections, dst_ids)
        dst_global_shift = _global_frame_shift(
            detections,
            by_frame.get(t, []),
            by_frame.get(t + 1, []),
            spatial_ndim,
        )
        matches = _global_point_matching(
            src_points,
            src_owner,
            dst_points,
            dst_owner,
            detections_by_id,
            config,
            spacing_scale=spacing_scale,
            dst_global_shift=dst_global_shift,
        )
        if not matches:
            return t, []
        return (
            t,
            _links_from_point_matches(
                detections_by_id,
                src_owner,
                dst_owner,
                matches,
                config,
                start_id=0,
            ),
        )

    workers = int(getattr(config, "tracking_workers", 0))
    if workers <= 0:
        workers = min(len(frames), max(1, min((os.cpu_count() or 1) - 1, 4)))
    workers = max(1, min(workers, len(frames) if frames else 1))

    if workers <= 1 or len(frames) <= 1:
        results = [build_for_frame(t) for t in frames]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(build_for_frame, frames))

    candidates: list[LinkCandidate] = []
    next_id = 0
    for _frame, local_links in sorted(results, key=lambda item: item[0]):
        for link in local_links:
            candidates.append(replace(link, id=next_id))
            next_id += 1
    return candidates


def _restore_unlinked_object_links(
    selected: set[int],
    pair_links: list[LinkCandidate],
    selected_context_links: list[LinkCandidate] | None = None,
) -> set[int]:
    if not pair_links:
        return selected

    selected_outgoing: set[int] = set()
    selected_incoming: set[int] = set()
    for link in selected_context_links or pair_links:
        if int(link.id) not in selected:
            continue
        selected_outgoing.add(int(link.src))
        selected_incoming.add(int(link.dst))

    for link in sorted(
        pair_links,
        key=lambda item: (
            float(item.cost),
            -float(item.matched_points),
            int(item.id),
        ),
    ):
        if int(link.id) in selected:
            continue
        src_id = int(link.src)
        dst_id = int(link.dst)
        if src_id in selected_outgoing and dst_id in selected_incoming:
            continue
        selected.add(int(link.id))
        selected_outgoing.add(src_id)
        selected_incoming.add(dst_id)
    return selected


def _restore_unlinked_large_target_links_from_small_sources(
    selected: set[int],
    pair_links: list[LinkCandidate],
    detections_by_id: dict[int, Detection],
    min_area: int,
    cost_cutoff: float,
) -> set[int]:
    if not pair_links:
        return selected

    selected_outgoing: set[int] = set()
    selected_incoming: set[int] = set()
    for link in pair_links:
        if int(link.id) not in selected:
            continue
        selected_outgoing.add(int(link.src))
        selected_incoming.add(int(link.dst))

    candidates = [
        link
        for link in pair_links
        if (
            int(detections_by_id[int(link.src)].area) <= min_area
            and int(detections_by_id[int(link.dst)].area) > min_area
            and int(link.dst) not in selected_incoming
        )
    ]
    if not candidates:
        return selected

    ordered_candidates = sorted(
        candidates,
        key=lambda item: (
            float(item.cost),
            -float(item.matched_points),
            int(item.id),
        ),
    )
    rescue_targets = {int(link.dst) for link in ordered_candidates}
    rescued_targets: set[int] = set()

    # A previously unlinked large target may receive multiple plausible small
    # sources. Keep each small source exclusive to its best available target.
    for link in ordered_candidates:
        if float(link.cost) > float(cost_cutoff):
            continue
        src_id = int(link.src)
        if src_id in selected_outgoing:
            continue
        selected.add(int(link.id))
        selected_outgoing.add(src_id)
        rescued_targets.add(int(link.dst))

    # Match the large-object fallback used above: if no candidate passed the
    # cutoff, retain the single best available small-source link.
    missing_targets = rescue_targets - rescued_targets
    for link in ordered_candidates:
        dst_id = int(link.dst)
        if dst_id not in missing_targets:
            continue
        src_id = int(link.src)
        if src_id in selected_outgoing:
            continue
        selected.add(int(link.id))
        selected_outgoing.add(src_id)
        rescued_targets.add(dst_id)
        missing_targets.remove(dst_id)
        if not missing_targets:
            break
    return selected


def _restore_unlinked_large_source_links_to_small_targets(
    selected: set[int],
    pair_links: list[LinkCandidate],
    detections_by_id: dict[int, Detection],
    min_area: int,
    cost_cutoff: float,
) -> set[int]:
    if not pair_links:
        return selected

    selected_outgoing: set[int] = set()
    selected_incoming: set[int] = set()
    for link in pair_links:
        if int(link.id) not in selected:
            continue
        selected_outgoing.add(int(link.src))
        selected_incoming.add(int(link.dst))

    candidates = [
        link
        for link in pair_links
        if (
            int(detections_by_id[int(link.src)].area) > min_area
            and int(detections_by_id[int(link.dst)].area) <= min_area
            and int(link.src) not in selected_outgoing
        )
    ]
    if not candidates:
        return selected

    ordered_candidates = sorted(
        candidates,
        key=lambda item: (
            float(item.cost),
            -float(item.matched_points),
            int(item.id),
        ),
    )
    rescue_sources = {int(link.src) for link in ordered_candidates}
    rescued_sources: set[int] = set()

    # A previously unlinked large source may divide into multiple plausible
    # small targets. Keep each small target exclusive to its best source.
    for link in ordered_candidates:
        if float(link.cost) > float(cost_cutoff):
            continue
        dst_id = int(link.dst)
        if dst_id in selected_incoming:
            continue
        selected.add(int(link.id))
        selected_incoming.add(dst_id)
        rescued_sources.add(int(link.src))

    # If no candidate passed the cutoff, retain the single best available
    # small-target link so a large source is not silently left without output.
    missing_sources = rescue_sources - rescued_sources
    for link in ordered_candidates:
        src_id = int(link.src)
        if src_id not in missing_sources:
            continue
        dst_id = int(link.dst)
        if dst_id in selected_incoming:
            continue
        selected.add(int(link.id))
        selected_incoming.add(dst_id)
        rescued_sources.add(src_id)
        missing_sources.remove(src_id)
        if not missing_sources:
            break
    return selected


def _select_links_for_frame_pair(
    pair_links: list[LinkCandidate],
    config: TrackingConfig,
    detections_by_id: dict[int, Detection],
    cost_cutoff: float,
) -> set[int]:
    min_area = int(config.min_link_size)
    # Use large-to-large links as the primary pool. Small objects remain
    # available only for the two directed rescue passes below.
    linkable_pairs = [
        link
        for link in pair_links
        if (
            int(detections_by_id[int(link.src)].area) > min_area
            and int(detections_by_id[int(link.dst)].area) > min_area
        )
    ]
    selected = {
        link.id
        for link in linkable_pairs
        if float(link.cost) <= float(cost_cutoff)
    }
    selected = _restore_unlinked_large_target_links_from_small_sources(
        selected,
        pair_links,
        detections_by_id,
        min_area,
        cost_cutoff,
    )
    selected = _restore_unlinked_large_source_links_to_small_targets(
        selected,
        pair_links,
        detections_by_id,
        min_area,
        cost_cutoff,
    )
    return _restore_unlinked_object_links(
        selected,
        linkable_pairs,
        selected_context_links=pair_links,
    )


def _select_links(
    detections: list[Detection],
    links: list[LinkCandidate],
    config: TrackingConfig,
) -> set[int]:
    by_frame_pair: dict[int, list[LinkCandidate]] = defaultdict(list)
    det_by_id = {det.id: det for det in detections}
    for link in links:
        by_frame_pair[det_by_id[link.src].frame].append(link)

    selected: set[int] = set()
    cost_cutoff = float(config.cost_cutoff)
    for frame in sorted(by_frame_pair):
        pair_links = by_frame_pair[frame]
        selected.update(_select_links_for_frame_pair(pair_links, config, det_by_id, cost_cutoff))
    return selected


def _linked_graph(
    detections: list[Detection],
    links: list[LinkCandidate],
    selected_link_ids: set[int],
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[tuple[int, int], LinkCandidate]]:
    incoming: dict[int, list[int]] = {det.id: [] for det in detections}
    outgoing: dict[int, list[int]] = {det.id: [] for det in detections}
    selected_by_pair: dict[tuple[int, int], LinkCandidate] = {}
    for link in links:
        if link.id not in selected_link_ids:
            continue
        outgoing[link.src].append(link.dst)
        incoming[link.dst].append(link.src)
        selected_by_pair[(link.src, link.dst)] = link
    return incoming, outgoing, selected_by_pair


def _assign_lineages(
    by_frame: dict[int, list[int]],
    incoming: dict[int, list[int]],
    selected_by_pair: dict[tuple[int, int], LinkCandidate],
) -> dict[int, int]:
    lineage_ids: dict[int, int] = {}
    next_lineage = 1
    for det_id in sorted(by_frame.get(0, [])):
        lineage_ids[det_id] = next_lineage
        next_lineage += 1

    max_frame = max(by_frame.keys(), default=-1)
    for frame in range(1, max_frame + 1):
        for det_id in sorted(by_frame.get(frame, [])):
            parents = incoming.get(det_id, [])
            if not parents:
                lineage_ids[det_id] = next_lineage
                next_lineage += 1
                continue
            dominant_parent = min(
                parents,
                key=lambda src_id: (
                    -selected_by_pair[(src_id, det_id)].matched_points,
                    selected_by_pair[(src_id, det_id)].cost,
                    src_id,
                ),
            )
            lineage_ids[det_id] = lineage_ids.get(dominant_parent, next_lineage)
            if lineage_ids[det_id] == next_lineage:
                next_lineage += 1
    return lineage_ids


def _component_nodes(
    sources: set[int],
    targets: set[int],
    outgoing: dict[int, list[int]],
    incoming: dict[int, list[int]],
) -> list[tuple[list[int], list[int]]]:
    components: list[tuple[list[int], list[int]]] = []
    visited_s: set[int] = set()
    visited_t: set[int] = set()

    for start in sorted(sources):
        if start in visited_s:
            continue
        comp_s: set[int] = set()
        comp_t: set[int] = set()
        queue: deque[tuple[str, int]] = deque([("s", start)])
        while queue:
            kind, node = queue.popleft()
            if kind == "s":
                if node in visited_s:
                    continue
                visited_s.add(node)
                comp_s.add(node)
                for dst in outgoing.get(node, []):
                    if dst in targets and dst not in visited_t:
                        queue.append(("t", dst))
            else:
                if node in visited_t:
                    continue
                visited_t.add(node)
                comp_t.add(node)
                for src in incoming.get(node, []):
                    if src in sources and src not in visited_s:
                        queue.append(("s", src))
        if comp_s or comp_t:
            components.append((sorted(comp_s), sorted(comp_t)))
    return components


def _relative_area_change(area_a: int, area_b: int) -> float:
    return abs(float(area_a) - float(area_b)) / max(float(area_a), float(area_b), 1.0)


def _classify_pair_event(
    src: int,
    dst: int,
    detections_by_id: dict[int, Detection],
    threshold: float,
    frame: int,
) -> TrackingEvent:
    src_det = detections_by_id[src]
    dst_det = detections_by_id[dst]
    change = _relative_area_change(src_det.area, dst_det.area)
    if change >= threshold:
        kind = "elongation" if dst_det.area > src_det.area else "shortening"
    else:
        kind = "continuation"
    return TrackingEvent(
        kind=kind,
        frame=frame,
        frame_from=frame,
        frame_to=frame + 1,
        sources=(src,),
        targets=(dst,),
    )


def _events_from_selected_links(
    detections: list[Detection],
    by_frame: dict[int, list[int]],
    selected_by_pair: dict[tuple[int, int], LinkCandidate],
    config: TrackingConfig,
    *,
    frame_pairs: set[tuple[int, int]] | None = None,
) -> list[TrackingEvent]:
    detections_by_id = {det.id: det for det in detections}
    events: list[TrackingEvent] = []
    covered_sources: set[int] = set()
    covered_targets: set[int] = set()
    threshold = float(config.event_delta_threshold)
    max_frame = max(by_frame.keys(), default=-1)
    if frame_pairs is None:
        source_frames = set(range(max_frame))
        target_frames = set(range(1, max_frame + 1))
    else:
        source_frames = {int(frame_from) for frame_from, _frame_to in frame_pairs}
        target_frames = {int(frame_to) for _frame_from, frame_to in frame_pairs}
    links_by_frame: dict[int, list[LinkCandidate]] = defaultdict(list)
    for (src, _dst), link in selected_by_pair.items():
        source = detections_by_id[int(src)]
        target = detections_by_id[int(link.dst)]
        pair = (int(source.frame), int(target.frame))
        if frame_pairs is None or pair in frame_pairs:
            links_by_frame[pair[0]].append(link)

    for frame in sorted(source_frames):
        if frame < 0 or frame >= max_frame or frame + 1 not in target_frames:
            continue
        pair_links = links_by_frame.get(frame, ())
        event_outgoing: dict[int, list[int]] = defaultdict(list)
        event_incoming: dict[int, list[int]] = defaultdict(list)
        for link in pair_links:
            src = int(link.src)
            dst = int(link.dst)
            event_outgoing[src].append(dst)
            event_incoming[dst].append(src)

        sources = {det_id for det_id in by_frame.get(frame, []) if event_outgoing.get(det_id)}
        targets = {det_id for det_id in by_frame.get(frame + 1, []) if event_incoming.get(det_id)}
        for comp_sources, comp_targets in _component_nodes(
            sources,
            targets,
            event_outgoing,
            event_incoming,
        ):
            src_ids = tuple(sorted(comp_sources))
            dst_ids = tuple(sorted(comp_targets))
            if len(src_ids) == 1 and len(dst_ids) == 1:
                events.append(
                    _classify_pair_event(
                        src_ids[0],
                        dst_ids[0],
                        detections_by_id,
                        threshold,
                        frame,
                    )
                )
            elif len(src_ids) == 1:
                events.append(
                    TrackingEvent("fission", frame, frame, frame + 1, src_ids, dst_ids)
                )
            elif len(dst_ids) == 1:
                events.append(
                    TrackingEvent("fusion", frame, frame, frame + 1, src_ids, dst_ids)
                )
            else:
                events.append(
                    TrackingEvent("split-merge", frame, frame, frame + 1, src_ids, dst_ids)
                )
            covered_sources.update(src_ids)
            covered_targets.update(dst_ids)

    for det in detections:
        if (
            det.frame in target_frames
            and det.id not in covered_targets
            and det.area <= int(config.min_link_size)
        ):
            events.append(
                TrackingEvent(
                    kind="birth",
                    frame=det.frame,
                    frame_from=det.frame,
                    frame_to=det.frame,
                    sources=(),
                    targets=(det.id,),
                )
            )
        if (
            det.frame in source_frames
            and det.id not in covered_sources
            and det.area <= int(config.min_link_size)
        ):
            events.append(
                TrackingEvent(
                    kind="death",
                    frame=det.frame,
                    frame_from=det.frame,
                    frame_to=det.frame,
                    sources=(det.id,),
                    targets=(),
                )
            )

    events.sort(key=lambda e: (e.frame_from, e.frame_to, e.kind, e.sources, e.targets))
    return events


def _tracked_labels_from_lineages(
    frame_labels: np.ndarray,
    detections: list[Detection],
    lineage_ids: dict[int, int],
) -> np.ndarray:
    tracked = np.zeros_like(frame_labels, dtype=np.int32)
    for det in detections:
        lineage_id = int(lineage_ids.get(det.id, 0))
        if lineage_id <= 0:
            continue
        target = tracked[(int(det.frame), *tuple(det.slice_tuple))]
        target[np.asarray(det.image, dtype=bool)] = lineage_id
    return tracked


def _tracking_result_with_selected_links(
    result: TrackingResult,
    config: TrackingConfig,
    selected_link_ids: set[int],
    *,
    message: str,
    rebuild_lineages: bool = True,
) -> TrackingResult:
    link_by_id = {int(link.id): link for link in result.links}
    selected_link_ids = {int(link_id) for link_id in selected_link_ids}
    unknown = selected_link_ids - set(link_by_id)
    if unknown:
        raise ValueError(f"Unknown tracking link id(s): {sorted(unknown)}")

    detections = list(result.detections)
    detections_by_id = {int(det.id): det for det in detections}
    by_frame: dict[int, list[int]] = defaultdict(list)
    for detection in detections:
        by_frame[int(detection.frame)].append(int(detection.id))
    selected_by_pair = {
        (int(link.src), int(link.dst)): link
        for link in result.links
        if int(link.id) in selected_link_ids
    }
    if rebuild_lineages:
        incoming, _outgoing, selected_by_pair = _linked_graph(
            detections,
            list(result.links),
            selected_link_ids,
        )
        lineage_ids = _assign_lineages(
            by_frame,
            incoming,
            selected_by_pair,
        )
        tracked_labels = _tracked_labels_from_lineages(
            result.frame_labels,
            detections,
            lineage_ids,
        )
    else:
        lineage_ids = result.lineage_ids
        tracked_labels = result.tracked_labels
    changed_link_ids = {
        int(link_id) for link_id in result.selected_link_ids
    } ^ selected_link_ids
    changed_frame_pairs = {
        (
            int(detections_by_id[int(link_by_id[link_id].src)].frame),
            int(detections_by_id[int(link_by_id[link_id].dst)].frame),
        )
        for link_id in changed_link_ids
    }
    can_update_events_incrementally = (
        bool(result.events)
        and bool(changed_frame_pairs)
        and all(
            frame_to == frame_from + 1
            for frame_from, frame_to in changed_frame_pairs
        )
    )
    if can_update_events_incrementally:
        events = [
            event
            for event in result.events
            if not any(
                (
                    event.kind == "birth"
                    and int(event.frame) == frame_to
                )
                or (
                    event.kind == "death"
                    and int(event.frame) == frame_from
                )
                or (
                    event.kind not in {"birth", "death"}
                    and int(event.frame_from) == frame_from
                    and int(event.frame_to) == frame_to
                )
                for frame_from, frame_to in changed_frame_pairs
            )
        ]
        events.extend(
            _events_from_selected_links(
                detections,
                by_frame,
                selected_by_pair,
                config,
                frame_pairs=changed_frame_pairs,
            )
        )
        events.sort(
            key=lambda event: (
                event.frame_from,
                event.frame_to,
                event.kind,
                event.sources,
                event.targets,
            )
        )
    else:
        events = _events_from_selected_links(
            detections,
            by_frame,
            selected_by_pair,
            config,
        )
    objective = float(
        sum(link_by_id[link_id].cost for link_id in selected_link_ids)
    )
    return replace(
        result,
        tracked_labels=tracked_labels,
        selected_link_ids=tuple(sorted(int(link_id) for link_id in selected_link_ids)),
        lineage_ids=lineage_ids,
        events=events,
        objective_value=objective,
        success=True,
        message=str(message),
    )


def set_tracking_link_selected(
    result: TrackingResult,
    config: TrackingConfig,
    link_id: int,
    *,
    selected: bool,
    rebuild_lineages: bool = True,
) -> TrackingResult:
    link_id = int(link_id)
    link_by_id = {int(link.id): link for link in result.links}
    if link_id not in link_by_id:
        raise ValueError(f"Unknown tracking candidate link id: {link_id}")

    selected_link_ids = {int(value) for value in result.selected_link_ids}
    if bool(selected):
        selected_link_ids.add(link_id)
        action = "added to"
    else:
        selected_link_ids.discard(link_id)
        action = "removed from"
    return _tracking_result_with_selected_links(
        result,
        config,
        selected_link_ids,
        message=f"Manual tracking refine: link {link_id} {action} the link pool.",
        rebuild_lineages=rebuild_lineages,
    )


def _manual_link_candidate(
    source: Detection,
    target: Detection,
    config: TrackingConfig,
    *,
    link_id: int,
    spacing: tuple[float, ...] | None,
) -> LinkCandidate:
    spatial_ndim = len(source.centroid)
    if len(target.centroid) != spatial_ndim:
        raise ValueError("Manual link detections must have matching dimensions.")
    spacing_scale = np.asarray(
        _normalize_spacing(spacing, spatial_ndim),
        dtype=float,
    )
    source_centroid = np.asarray(source.centroid, dtype=float) * spacing_scale
    target_centroid = np.asarray(target.centroid, dtype=float) * spacing_scale
    centroid_distance = float(
        np.linalg.norm(target_centroid - source_centroid)
    )

    source_points = np.asarray(source.sampled_coords, dtype=float)
    target_points = np.asarray(target.sampled_coords, dtype=float)
    if source_points.size == 0:
        source_points = np.asarray(source.coords, dtype=float)
    if target_points.size == 0:
        target_points = np.asarray(target.coords, dtype=float)

    if source_points.size > 0 and target_points.size > 0:
        source_metric = source_points * spacing_scale[None, :]
        target_metric = target_points * spacing_scale[None, :]
        source_aligned = (
            source_metric
            + (
                np.mean(target_metric, axis=0)
                - np.mean(source_metric, axis=0)
            )[None, :]
        )
        source_distances = np.asarray(
            cKDTree(target_metric).query(source_aligned, k=1)[0],
            dtype=float,
        )
        target_distances = np.asarray(
            cKDTree(source_aligned).query(target_metric, k=1)[0],
            dtype=float,
        )
        match_radius = max(float(np.min(spacing_scale)), 1e-6)
        source_fraction = float(
            np.mean(source_distances <= match_radius)
        )
        target_fraction = float(
            np.mean(target_distances <= match_radius)
        )
        overlap_like = 0.5 * (source_fraction + target_fraction)
        all_distances = np.concatenate(
            (source_distances, target_distances)
        )
        point_distance = float(np.min(all_distances))
        average_point_distance = float(np.mean(all_distances))
        matched_points = 0.5 * (
            float(np.count_nonzero(source_distances <= match_radius))
            + float(np.count_nonzero(target_distances <= match_radius))
        )
    else:
        source_fraction = 0.0
        target_fraction = 0.0
        overlap_like = 0.0
        point_distance = centroid_distance
        average_point_distance = centroid_distance
        matched_points = 0.0

    distance_norm = float(config.max_distance)
    if not np.isfinite(distance_norm):
        raw_distance_cost = 0.0
    else:
        raw_distance_cost = centroid_distance / max(distance_norm, 1e-6)
    raw_overlap_cost = 1.0 - overlap_like
    point_support_capacity = max(
        min(
            len(source_points),
            len(target_points),
            max(int(config.point_support_saturation), 1),
        ),
        1,
    )
    raw_point_support_cost = 1.0 - min(
        matched_points / float(point_support_capacity),
        1.0,
    )
    distance_term = float(config.distance_weight) * raw_distance_cost
    overlap_term = float(config.overlap_weight) * raw_overlap_cost
    point_support_term = (
        float(config.point_support_weight) * raw_point_support_cost
    )
    area_ratio = abs(int(source.area) - int(target.area)) / max(
        int(source.area),
        int(target.area),
        1,
    )
    return LinkCandidate(
        id=int(link_id),
        src=int(source.id),
        dst=int(target.id),
        cost=float(distance_term + overlap_term + point_support_term),
        raw_distance_cost=float(raw_distance_cost),
        raw_overlap_cost=float(raw_overlap_cost),
        raw_point_support_cost=float(raw_point_support_cost),
        distance_term=float(distance_term),
        overlap_term=float(overlap_term),
        point_support_term=float(point_support_term),
        distance=float(centroid_distance),
        overlap=float(overlap_like),
        area_ratio=float(area_ratio),
        point_distance=float(point_distance),
        average_point_distance=float(average_point_distance),
        matched_points=float(matched_points),
        src_fraction=float(source_fraction),
        dst_fraction=float(target_fraction),
    )


def ensure_manual_tracking_link_candidate(
    result: TrackingResult,
    config: TrackingConfig,
    source_detection_id: int,
    target_detection_id: int,
    *,
    spacing: tuple[float, ...] | None = None,
) -> tuple[TrackingResult, int, bool]:
    """Ensure an explicit adjacent-frame pair exists in the candidate pool."""
    detections_by_id = {
        int(detection.id): detection
        for detection in result.detections
    }
    source_id = int(source_detection_id)
    target_id = int(target_detection_id)
    source = detections_by_id.get(source_id)
    target = detections_by_id.get(target_id)
    if source is None:
        raise ValueError(f"Unknown tracking source detection id: {source_id}")
    if target is None:
        raise ValueError(f"Unknown tracking target detection id: {target_id}")
    if int(target.frame) != int(source.frame) + 1:
        raise ValueError(
            "Manual tracking links must connect adjacent frames in forward time."
        )

    existing = next(
        (
            link
            for link in result.links
            if int(link.src) == source_id and int(link.dst) == target_id
        ),
        None,
    )
    if existing is not None:
        return result, int(existing.id), False

    link_id = max((int(link.id) for link in result.links), default=-1) + 1
    link = _manual_link_candidate(
        source,
        target,
        config,
        link_id=link_id,
        spacing=spacing,
    )
    result_with_candidate = replace(
        result,
        links=[*result.links, link],
    )
    return result_with_candidate, int(link_id), True


def add_manual_tracking_link(
    result: TrackingResult,
    config: TrackingConfig,
    source_detection_id: int,
    target_detection_id: int,
    *,
    spacing: tuple[float, ...] | None = None,
    rebuild_lineages: bool = True,
) -> tuple[TrackingResult, int]:
    """Create or select an explicit adjacent-frame link chosen in the viewer."""
    result_with_candidate, link_id, _created = (
        ensure_manual_tracking_link_candidate(
            result,
            config,
            source_detection_id,
            target_detection_id,
            spacing=spacing,
        )
    )
    if int(link_id) in {
        int(value) for value in result_with_candidate.selected_link_ids
    }:
        return result_with_candidate, int(link_id)
    selected_link_ids = {
        int(value) for value in result_with_candidate.selected_link_ids
    }
    selected_link_ids.add(int(link_id))
    refined = _tracking_result_with_selected_links(
        result_with_candidate,
        config,
        selected_link_ids,
        message=(
            "Manual tracking refine: viewer-selected link "
            f"{int(source_detection_id)}->{int(target_detection_id)} "
            "added to the link pool."
        ),
        rebuild_lineages=rebuild_lineages,
    )
    return refined, int(link_id)


def restore_tracking_link_selection(
    result: TrackingResult,
    config: TrackingConfig,
    selected_link_ids: tuple[int, ...] | list[int] | set[int],
    *,
    message: str = "Manual tracking refine undone.",
    rebuild_lineages: bool = True,
) -> TrackingResult:
    """Rebuild a tracking result from a previously saved link selection."""
    return _tracking_result_with_selected_links(
        result,
        config,
        {int(link_id) for link_id in selected_link_ids},
        message=message,
        rebuild_lineages=rebuild_lineages,
    )


def rebuild_tracking_lineages(result: TrackingResult) -> TrackingResult:
    """Materialize lineage ids and tracked labels for the current link pool."""
    detections = list(result.detections)
    by_frame: dict[int, list[int]] = defaultdict(list)
    for detection in detections:
        by_frame[int(detection.frame)].append(int(detection.id))
    incoming, _outgoing, selected_by_pair = _linked_graph(
        detections,
        list(result.links),
        {int(link_id) for link_id in result.selected_link_ids},
    )
    lineage_ids = _assign_lineages(
        by_frame,
        incoming,
        selected_by_pair,
    )
    tracked_labels = _tracked_labels_from_lineages(
        result.frame_labels,
        detections,
        lineage_ids,
    )
    return replace(
        result,
        tracked_labels=tracked_labels,
        lineage_ids=lineage_ids,
    )


def track_segmentations_ilp(
    segmentation: np.ndarray,
    spacing: tuple[float, ...] | None = None,
    config: TrackingConfig | None = None,
    intensity_image: np.ndarray | None = None,
    cancel_check: callable | None = None,
) -> TrackingResult:
    config = config or TrackingConfig()
    segmentation = np.asarray(segmentation)
    spatial_ndim = _spatial_ndim(segmentation)
    spacing_scale = _spacing_array(_normalize_spacing(spacing, spatial_ndim), spatial_ndim)
    if intensity_image is not None:
        intensity_image = np.asarray(intensity_image)
        if intensity_image.shape != segmentation.shape:
            raise ValueError(
                "intensity_image must match segmentation shape, got "
                f"{intensity_image.shape} and {segmentation.shape}."
            )

    frame_labels = _label_stack(segmentation)
    detections, by_frame = _extract_detections(
        frame_labels,
        config,
        intensity_image=intensity_image,
        cancel_check=cancel_check,
    )
    if not detections:
        empty = np.zeros_like(frame_labels, dtype=np.int32)
        return TrackingResult(
            frame_labels=frame_labels,
            tracked_labels=empty,
            detections=[],
            links=[],
            selected_link_ids=(),
            lineage_ids={},
            events=[],
            objective_value=0.0,
            success=True,
            message="No detections.",
        )

    if cancel_check is not None and cancel_check():
        raise TrackingCancelledError("Tracking cancelled.")
    links = _build_link_candidates(
        detections,
        by_frame,
        config,
        spacing_scale=spacing_scale,
        cancel_check=cancel_check,
    )
    if cancel_check is not None and cancel_check():
        raise TrackingCancelledError("Tracking cancelled.")
    selected_link_ids = _select_links(detections, links, config)
    incoming, outgoing, selected_by_pair = _linked_graph(detections, links, selected_link_ids)
    if cancel_check is not None and cancel_check():
        raise TrackingCancelledError("Tracking cancelled.")
    lineage_ids = _assign_lineages(by_frame, incoming, selected_by_pair)
    tracked_labels = _tracked_labels_from_lineages(frame_labels, detections, lineage_ids)
    if cancel_check is not None and cancel_check():
        raise TrackingCancelledError("Tracking cancelled.")
    events = _events_from_selected_links(
        detections,
        by_frame,
        selected_by_pair,
        config,
    )
    objective = float(sum(link.cost for link in links if link.id in selected_link_ids))

    return TrackingResult(
        frame_labels=frame_labels,
        tracked_labels=tracked_labels,
        detections=detections,
        links=links,
        selected_link_ids=tuple(sorted(selected_link_ids)),
        lineage_ids=lineage_ids,
        events=events,
        objective_value=objective,
        success=True,
        message="Global interior point matching completed. [tracking-global-point-v1]",
    )
