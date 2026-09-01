from __future__ import annotations

import numpy as np


_UNIQUE_CHUNK_SIZE = 262_144


def limited_unique_values(
    data,
    max_values: int,
    *,
    chunk_size: int = _UNIQUE_CHUNK_SIZE,
) -> np.ndarray | None:
    """Return unique values, or ``None`` as soon as the limit is exceeded.

    Candidate-layer checks run on the Qt thread, so sorting an entire TZYX
    stack here can freeze the viewer. Processing bounded flat chunks preserves
    the exact result while allowing raw intensity images to be rejected after
    the first chunk in the usual case.
    """
    if max_values < 1:
        raise ValueError("max_values must be at least 1")
    chunk_size = max(1, int(chunk_size))

    size = getattr(data, "size", None)
    if size is None:
        data = np.asarray(data)
        size = data.size
    size = int(size)
    if size == 0:
        return np.asarray([], dtype=getattr(data, "dtype", float))

    try:
        flat = data.reshape(-1)
    except (AttributeError, TypeError, ValueError):
        flat = np.asarray(data).reshape(-1)

    merged = None
    for start in range(0, size, chunk_size):
        chunk = np.asarray(flat[start : min(start + chunk_size, size)])
        chunk_unique = np.unique(chunk)
        if merged is None:
            merged = chunk_unique
        else:
            merged = np.unique(np.concatenate((merged, chunk_unique)))
        if merged.size > max_values:
            return None

    return merged


def is_binary_mask_data(data) -> bool:
    """Return whether data has one/two values and includes background zero."""
    unique = limited_unique_values(data, 2)
    if unique is None or unique.size == 0:
        return False
    if unique.size == 1:
        return bool(unique[0] == 0)
    return bool(np.any(unique == 0))
