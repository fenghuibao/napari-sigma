"""Shared, UI-independent layer metadata conventions."""
from __future__ import annotations

import numpy as np

# Keys the writer persists inside a TIFF's sigma_metadata block, and that the
# reader must therefore restore. Only SIGMA's own writer produces that block,
# so recovering these is honouring what the file states about itself, not
# inferring a role for an arbitrary image.
SIGMA_TIFF_METADATA_KEYS = (
    "sigma_layer_role",
    "sigma_saved_structure",
    "is_frangi",
    # Without this a saved mask loses its role on reload and is then offered
    # as *raw* input, and appears in both proximity dropdowns at once.
    "is_segmentation",
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
    # 2D data has only two Hessian eigenvalues, so a "sheetness" response is
    # really vesselness. Persist that caveat with the layer.
    "frangi_sheetness_substituted",
    "is_proximity_roi_mask",
    "proximity_source_layer",
    "proximity_roi_count",
)


# Metadata keys that mark a layer as an analysis overlay rather than data the
# user can choose as an input. Single definition: an earlier copy was
# open-coded in two proximity predicates, so a fourth key added centrally
# would have left analysis overlays in both proximity dropdowns.
ANALYSIS_AUX_METADATA_KEYS = (
    "is_analysis_labels",
    "is_analysis_highlight",
    "is_analysis_topology",
)


def is_analysis_aux_metadata(metadata: dict | None) -> bool:
    md = metadata or {}
    return any(md.get(key) for key in ANALYSIS_AUX_METADATA_KEYS)


def axes_for_ndim(ndim: int) -> str:
    """Canonical axis tag for an array of this rank.

    Single definition so the writer's tag and any tag stamped onto a layer
    cannot disagree; an earlier duplicate omitted the 5-D entry and produced
    "VWXYZ" where the writer produced "TCZYX".
    """
    return {
        2: "YX",
        3: "ZYX",
        4: "TZYX",
        5: "TCZYX",
    }.get(ndim, "".join("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[-ndim:]))


def dims_tag_from_metadata(metadata: dict | None) -> str:
    """Axis tag for this metadata, resolving ``dims_out`` before ``dims``.

    ``dims_out`` describes the array as it now exists and ``dims`` can be an
    inherited leftover: a derived layer that squeezed time keeps
    ``dims="TZYX"`` while its data is ZYX. Callers holding only a metadata
    dict must use this, not their own ordering, or a 3-D volume gets treated
    as a time series (see tests/test_io_regressions.py).
    """
    md = metadata or {}
    return str(md.get("dims_out") or md.get("dims") or md.get("inferred_dims") or "").upper()


def layer_dims_tag(layer) -> str:
    return dims_tag_from_metadata(getattr(layer, "metadata", {}))


def unit_from_metadata(metadata: dict | None) -> str:
    md = metadata or {}
    return md.get("PhysicalSizeXUnit") or md.get("unit") or md.get("Units") or "um"


def measurement_unit_for_layer(layer) -> str:
    """The unit measurements from this layer are actually in.

    ``unit_from_metadata`` defaults to "um" so the scale bar always has
    something to draw, but the magnitudes come from ``layer.scale``, which
    defaults to 1.0. A layer that declares no unit and carries an identity
    scale is therefore uncalibrated, and its numbers are voxel counts — saying
    "um" there labels raw counts as micrometres.
    """
    md = getattr(layer, "metadata", {}) or {}
    declared = md.get("PhysicalSizeXUnit") or md.get("unit") or md.get("Units")
    if declared:
        return str(declared)
    scale = getattr(layer, "scale", None)
    try:
        values = [float(value) for value in scale]
    except (TypeError, ValueError):
        return "um"
    if values and all(value == 1.0 for value in values):
        return "px"
    return "um"


def squeeze_leading_singletons(arr: np.ndarray, target_ndim: int) -> np.ndarray:
    out = np.asarray(arr)
    while out.ndim > target_ndim and out.shape[0] == 1:
        out = out[0]
    return out
