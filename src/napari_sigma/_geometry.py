"""Physical measures of exposed pixel/voxel faces (no GUI dependencies)."""
from __future__ import annotations

import numpy as np


def exposed_face_measure(mask, spacing, selection=None) -> float:
    binary = np.asarray(mask, dtype=bool)
    sampling = np.asarray(spacing, dtype=float)
    if binary.ndim not in (2, 3) or sampling.shape != (binary.ndim,):
        raise ValueError("A 2D/3D mask needs one spacing per spatial axis.")
    if not np.all(np.isfinite(sampling)) or np.any(sampling <= 0):
        raise ValueError("Spacing must be finite and positive.")
    chosen = binary if selection is None else np.asarray(selection, dtype=bool)
    if chosen.shape != binary.shape:
        raise ValueError("Surface selection must match the object mask shape.")
    chosen = chosen & binary
    total = 0.0
    for axis in range(binary.ndim):
        face_area = float(np.prod(np.delete(sampling, axis)))
        current = [slice(None)] * binary.ndim
        neighbor = [slice(None)] * binary.ndim
        current[axis], neighbor[axis] = slice(1, None), slice(None, -1)
        before = chosen.copy()
        before[tuple(current)] &= ~binary[tuple(neighbor)]
        after = chosen.copy()
        after[tuple(neighbor)] &= ~binary[tuple(current)]
        total += (np.count_nonzero(before) + np.count_nonzero(after)) * face_area
    return float(total)
