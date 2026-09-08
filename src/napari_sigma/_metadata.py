"""Shared, UI-independent layer metadata conventions."""
from __future__ import annotations

import numpy as np


def layer_dims_tag(layer) -> str:
    md = getattr(layer, "metadata", {}) or {}
    return str(md.get("dims_out") or md.get("dims") or md.get("inferred_dims") or "").upper()


def unit_from_metadata(metadata: dict | None) -> str:
    md = metadata or {}
    return md.get("PhysicalSizeXUnit") or md.get("unit") or md.get("Units") or "um"


def squeeze_leading_singletons(arr: np.ndarray, target_ndim: int) -> np.ndarray:
    out = np.asarray(arr)
    while out.ndim > target_ndim and out.shape[0] == 1:
        out = out[0]
    return out
