# Author: Huibao Feng
# Date: 2025-11-02

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["SegmentationInfo", "segmentation"]

# ---------------------
# Public datatypes
# ---------------------

@dataclass
class SegmentationInfo:
    """Holds optimization traces for UI/preview."""
    iterations_run: int
    deltas: list[float]
    converged: bool
    state: dict[str, np.ndarray] | None = None


# ---------------------
# Math helpers
# ---------------------

_TINY = 1e-15
_DEFAULT_TOL = 1e-8
_GMM_INIT_MAX_POINTS = 100_000
_EM_CONVERGENCE_MONITOR_POINTS = 65_536
_DEFAULT_EM_CLASS_SAMPLE_POINTS = 1_000_000
_MPS_OTSU_MONITOR_POINTS = 524_288

# Floor added to `f0` inside `log(...)` for the Frangi anchor potential.
# `1e-15` makes the anchor saturate at ±exp(34.5) ≈ 1e15 at the extremes,
# i.e. the Frangi term dominates GMM/smoothness whenever the Frangi response
# is very high or very low. Raising the floor caps anchor strength and gives
# the rest of the MRF more say, but in practice this oversmoothes when the
# Frangi response correctly identifies structure — we'd rather trust Frangi
# at the extremes. Adjust here if you ever want to soften the anchor again.
# Applies identically to 2D and 3D paths.
_FRANGI_ANCHOR_FLOOR = 1e-15


def _as_tensor_without_readonly_alias(
    value: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Convert ``value`` without aliasing read-only NumPy storage.

    TIFF readers can expose memory-mapped/read-only NumPy arrays. PyTorch
    warns when ``as_tensor`` aliases those arrays because a later tensor write
    would have undefined behaviour. Copy only that case; retain the normal
    zero-copy path for writable arrays and existing tensors.
    """
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        return torch.tensor(value, device=device)
    return torch.as_tensor(value, device=device)


def _gaussian(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Element-wise Gaussian pdf."""
    return 1.0 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(-(x - mu) ** 2 / (2 * sigma**2))


def _threshold_otsu_torch(data: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """Compute an Otsu threshold on the input tensor's current device.

    Segmentation normalizes ``data`` to [0, 255] before this function is
    called, so fixed histogram bounds avoid reading min/max back to the host.
    """
    bins = max(int(bins), 2)
    histogram_data = data.float().reshape(-1)
    if data.device.type == "mps" and histogram_data.numel() > _MPS_OTSU_MONITOR_POINTS:
        stride = max(
            (histogram_data.numel() + _MPS_OTSU_MONITOR_POINTS - 1)
            // _MPS_OTSU_MONITOR_POINTS,
            1,
        )
        histogram_data = histogram_data[::stride][:_MPS_OTSU_MONITOR_POINTS]
    counts = torch.histc(histogram_data, bins=bins, min=0.0, max=255.0)
    centers = (
        torch.arange(bins, dtype=counts.dtype, device=counts.device) + 0.5
    ) * (255.0 / bins)
    weighted = counts * centers
    weight_low = torch.cumsum(counts, dim=0)
    weight_high = torch.flip(torch.cumsum(torch.flip(counts, dims=(0,)), dim=0), dims=(0,))
    mean_low = torch.cumsum(weighted, dim=0) / weight_low.clamp_min(1.0)
    mean_high = torch.flip(
        torch.cumsum(torch.flip(weighted, dims=(0,)), dim=0),
        dims=(0,),
    ) / weight_high.clamp_min(1.0)
    valid = (weight_low[:-1] > 0) & (weight_high[1:] > 0)
    between_class_variance = (
        weight_low[:-1]
        * weight_high[1:]
        * (mean_low[:-1] - mean_high[1:]).square()
    )
    between_class_variance = torch.where(
        valid,
        between_class_variance,
        torch.full_like(between_class_variance, -1.0),
    )
    return centers[torch.argmax(between_class_variance)]


def _binary_label_from_likelihood(class_likelihood: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Return binary MAP labels without the slow general MPS argmax kernel."""
    return (class_likelihood[1] > class_likelihood[0]).reshape(shape).to(dtype=torch.uint8)


def _fill_gmm_class_likelihood(
    output: torch.Tensor,
    data: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    pairwise_potential: torch.Tensor,
    *,
    consume_pairwise: bool = False,
    log_output: bool = False,
) -> None:
    """Accumulate GMM components without a component-by-volume broadcast."""
    # MPS can drop in-place additions into a nonzero-offset channel view after
    # the preceding channel has been written. Accumulate that channel in an
    # independent tensor, then copy the finished scores into the requested
    # output view.
    use_mps_view_workaround = (
        log_output
        and output.device.type == "mps"
        and int(output.storage_offset()) != 0
    )
    accumulator = torch.zeros_like(output) if use_mps_view_workaround else output
    if accumulator is output:
        accumulator.zero_()
    for component in range(int(pi.numel())):
        component_likelihood = _gaussian(data, mu[component], sigma[component]).squeeze(0)
        accumulator.add_(pi[component] * component_likelihood)
    if log_output:
        # Keep the clamp/log/subtract transformation out-of-place as well:
        # chained in-place view operations have the same MPSGraph alias issue.
        log_scores = torch.log(torch.clamp(accumulator, min=_TINY))
        output.copy_(log_scores - pairwise_potential)
        return
    if consume_pairwise:
        attenuation = pairwise_potential.neg_().exp_()
    else:
        attenuation = torch.exp(-pairwise_potential)
    output.mul_(attenuation)


def _calculate_resp(X: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """E-step responsibilities for a 1D GMM."""
    resp = pi * _gaussian(X.reshape(-1, 1), mu, sigma)
    return resp / (resp.sum(dim=1, keepdim=True) + _TINY)


def _loglh_gmm(X: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Log-likelihood of a 1D GMM."""
    return torch.log((pi * _gaussian(X.reshape(-1, 1), mu, sigma)).sum(dim=1) + _TINY).sum()


def _calculate_resp_masked(
    X_col: torch.Tensor,
    mask: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Responsibilities on fixed-size data with a binary sample mask."""
    resp = pi * _gaussian(X_col, mu, sigma)
    resp = resp / (resp.sum(dim=1, keepdim=True) + _TINY)
    return resp * mask


def _loglh_gmm_masked(
    X_col: torch.Tensor,
    mask_flat: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Masked log-likelihood on fixed-size data."""
    loglh = torch.log((pi * _gaussian(X_col, mu, sigma)).sum(dim=1) + _TINY)
    return (loglh * mask_flat).sum()


# ---------------------
# Pairwise & potentials
# ---------------------

def _accumulate_axis_pairwise(
    accumulator: torch.Tensor,
    label: torch.Tensor,
    axis: int,
    smoothness: float,
) -> None:
    """Add both neighbors on one axis without padded full-volume copies."""
    ndim = label.ndim

    # Negative-direction neighbor. The padded boundary has label 0, whose
    # class-0 potential is -smoothness under the original formulation.
    boundary = [slice(None)] * ndim
    boundary[axis] = 0
    accumulator[tuple(boundary)].sub_(smoothness)
    if label.shape[axis] > 1:
        destination = [slice(None)] * ndim
        source = [slice(None)] * ndim
        destination[axis] = slice(1, None)
        source[axis] = slice(None, -1)
        accumulator[tuple(destination)].add_(label[tuple(source)], alpha=2.0 * smoothness)
        accumulator[tuple(destination)].sub_(smoothness)

    # Positive-direction neighbor, with the same zero-label boundary rule.
    boundary[axis] = -1
    accumulator[tuple(boundary)].sub_(smoothness)
    if label.shape[axis] > 1:
        destination[axis] = slice(None, -1)
        source[axis] = slice(1, None)
        accumulator[tuple(destination)].add_(label[tuple(source)], alpha=2.0 * smoothness)
        accumulator[tuple(destination)].sub_(smoothness)


def _frangi_potential(frangi: torch.Tensor, beta2: float):
    f0 = frangi / (frangi.max() + _TINY)
    f1 = 1.0 - f0
    # Use `_FRANGI_ANCHOR_FLOOR` (module-level) as the log floor so the anchor
    # has bounded strength and the rest of the MRF (GMM, smoothness, Frangi-
    # PSF correction) can actually contribute to the decision.
    return (
        beta2 * torch.log(f0 + _FRANGI_ANCHOR_FLOOR),
        beta2 * torch.log(f1 + _FRANGI_ANCHOR_FLOOR),
    )


def _pairwise_potential(
    image_label: torch.Tensor,
    beta1: float,
    sigma: float,
    frangi_pot0: torch.Tensor,
    frangi_pot1: torch.Tensor,
    device: torch.device,
    pixel_size_xy: float,
    pixel_size_z: float | None = None,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return unary+pairwise potentials for both classes stacked along first dim."""
    # Keep the Ising/Potts prior in voxel-neighbor units: beta1 is the xy
    # smoothness strength. Do not add a distance-decay term here; physical
    # sampling anisotropy is handled explicitly for z neighbors below.
    _ = sigma  # kept for API compatibility with older call sites
    smooth_xy = float(beta1)
    expected_shape = (2, *image_label.shape[1:])
    if output is None:
        output = torch.empty(expected_shape, dtype=frangi_pot0.dtype, device=device)
    elif tuple(output.shape) != expected_shape:
        raise ValueError(f"output shape {tuple(output.shape)} does not match {expected_shape}")

    # Build class 0 directly in the reusable output buffer. Keeping the channel
    # view makes its shape match `image_label` for the axis accumulator.
    pairwise_class0 = output[0:1]
    # MPSGraph cannot reliably lower float accumulator.add_(uint8, alpha=float):
    # it creates an incompatible float-volume × uint8-scalar multiply. Reuse
    # the still-free class-1 buffer as a float label workspace.
    pairwise_label = output[1:2]
    pairwise_label.copy_(image_label)
    pairwise_class0.zero_()
    if image_label.ndim == 3:
        _accumulate_axis_pairwise(pairwise_class0, pairwise_label, 1, smooth_xy)
        _accumulate_axis_pairwise(pairwise_class0, pairwise_label, 2, smooth_xy)
    else:
        assert pixel_size_z is not None, "pixel_size_z required for 3D volumes"
        # 3D Ising coupling should not treat adjacent z-slices as equally close
        # as adjacent xy pixels when dz > dxy. Use only the explicit z/xy
        # sampling anisotropy multiplier. PSF correction belongs mainly in the
        # Frangi evidence term, not the shape prior.
        z_anisotropy = min(1.0, float(pixel_size_xy) / max(float(pixel_size_z), _TINY))
        smooth_z = float(beta1 * z_anisotropy)
        _accumulate_axis_pairwise(pairwise_class0, pairwise_label, 1, smooth_z)
        _accumulate_axis_pairwise(pairwise_class0, pairwise_label, 2, smooth_xy)
        _accumulate_axis_pairwise(pairwise_class0, pairwise_label, 3, smooth_xy)

    # For binary labels, the original class-1 indicator is exactly the
    # negative of the class-0 indicator. Only one pairwise volume is needed.
    output[1].copy_(pairwise_class0[0]).neg_().add_(frangi_pot1[0])
    output[0].add_(frangi_pot0[0])
    return output


# ---------------------
# GMM parameter flows
# ---------------------

def _parameter_initialization(
    device: torch.device,
    pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
    data_fore: torch.Tensor, data_back: torch.Tensor,
    n_fore: int, n_back: int,
    random_state: int | None = None,
):
    """KMeans init mixed with wide sigma, uniform pi."""
    from sklearn.cluster import KMeans  # lazy import to keep the top clean

    data_back = data_back.reshape(-1)
    data_fore = data_fore.reshape(-1)
    if data_back.numel() > _GMM_INIT_MAX_POINTS:
        indices = torch.randint(data_back.numel(), (_GMM_INIT_MAX_POINTS,), device=data_back.device)
        data_back = data_back[indices]
    if data_fore.numel() > _GMM_INIT_MAX_POINTS:
        indices = torch.randint(data_fore.numel(), (_GMM_INIT_MAX_POINTS,), device=data_fore.device)
        data_fore = data_fore[indices]

    # sklearn is intentionally retained here: tested torch-native initializers
    # changed the downstream mask. Copy each class once and reuse that array
    # for both the unique-value guard and KMeans fitting.
    data_back_cpu = data_back.detach().cpu().reshape(-1).numpy()
    data_fore_cpu = data_fore.detach().cpu().reshape(-1).numpy()

    def _effective_cluster_count(values: np.ndarray, requested: int) -> int:
        if values.size < 1:
            return 0
        return min(int(requested), int(values.size), int(np.unique(values).size))

    n_back_eff = _effective_cluster_count(data_back_cpu, n_back)
    n_fore_eff = _effective_cluster_count(data_fore_cpu, n_fore)
    if n_back_eff < 1 or n_fore_eff < 1:
        raise ValueError(
            "Not enough foreground/background samples for GMM initialization; "
            "try lowering the component counts."
        )

    def _kmeans_centers(values: np.ndarray, n_clusters: int) -> torch.Tensor:
        centers = KMeans(
            n_clusters=n_clusters,
            n_init="auto",
            random_state=random_state,
        ).fit(
            values.reshape(-1, 1)
        ).cluster_centers_.reshape(-1)
        return torch.tensor(centers, dtype=torch.float32, device=device)

    bg_centers = _kmeans_centers(data_back_cpu, n_back_eff)
    fg_centers = _kmeans_centers(data_fore_cpu, n_fore_eff)
    mu[:n_back_eff, 0] = bg_centers
    mu[n_back_eff:n_back, 0] = bg_centers[-1]
    mu[:n_fore_eff, 1] = fg_centers
    mu[n_fore_eff:n_fore, 1] = fg_centers[-1]
    sigma[:n_back, 0] = 256
    sigma[:n_fore, 1] = 256
    pi[:n_back, 0] = 0.0
    pi[:n_fore, 1] = 0.0
    pi[:n_back_eff, 0] = 1.0 / n_back_eff
    pi[:n_fore_eff, 1] = 1.0 / n_fore_eff
    return pi, mu, sigma


def _switch_parameters(
    pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
    label: torch.Tensor, n_fore: int, n_back: int
):
    """Swap min-foreground and max-background components if their means cross."""
    min_fore = torch.min(mu[:n_fore, 1])
    max_back = torch.max(mu[:n_back, 0])
    min_fore_idx = int(torch.argmin(mu[:n_fore, 1]))
    max_back_idx = int(torch.argmax(mu[:n_back, 0]))

    mu[min_fore_idx, 1] = max_back
    mu[max_back_idx, 0] = min_fore

    n_fg = int((label == 1).sum())
    n_bg = int((label == 0).sum())

    pi_fg = pi[min_fore_idx, 1].clone()
    pi_bg = pi[max_back_idx, 0].clone()

    # avoid zero-div
    n_fg = max(n_fg, 1)
    n_bg = max(n_bg, 1)

    pi[min_fore_idx, 1] = pi_bg * (n_bg / n_fg)
    pi[max_back_idx, 0] = pi_fg * (n_fg / n_bg)
    pi /= pi.sum(dim=0, keepdim=True) + _TINY

    sigma_fg = sigma[min_fore_idx, 1].clone()
    sigma_bg = sigma[max_back_idx, 0].clone()
    sigma[min_fore_idx, 1] = sigma_bg
    sigma[max_back_idx, 0] = sigma_fg
    return pi, mu, sigma


def _switch_parameters_device(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    n_fore: int,
    n_back: int,
    foreground_count: int,
    background_count: int,
):
    """Conditionally swap crossed components without leaving the device."""
    min_fore, min_fore_idx = torch.min(mu[:n_fore, 1], dim=0)
    max_back, max_back_idx = torch.max(mu[:n_back, 0], dim=0)
    should_switch = min_fore < max_back

    candidate_pi = pi.clone()
    candidate_mu = mu.clone()
    candidate_sigma = sigma.clone()
    candidate_mu[min_fore_idx, 1] = max_back
    candidate_mu[max_back_idx, 0] = min_fore

    n_fg = max(int(foreground_count), 1)
    n_bg = max(int(background_count), 1)
    pi_fg = pi[min_fore_idx, 1]
    pi_bg = pi[max_back_idx, 0]
    candidate_pi[min_fore_idx, 1] = pi_bg * (n_bg / n_fg)
    candidate_pi[max_back_idx, 0] = pi_fg * (n_fg / n_bg)
    candidate_pi = candidate_pi / (candidate_pi.sum(dim=0, keepdim=True) + _TINY)

    candidate_sigma[min_fore_idx, 1] = sigma[max_back_idx, 0]
    candidate_sigma[max_back_idx, 0] = sigma[min_fore_idx, 1]
    return (
        torch.where(should_switch, candidate_pi, pi),
        torch.where(should_switch, candidate_mu, mu),
        torch.where(should_switch, candidate_sigma, sigma),
    )


def _em_once(
    data_flat: torch.Tensor,
    mask_fore_flat: torch.Tensor,
    mask_back_flat: torch.Tensor,
    n_fore: int, n_back: int,
    pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
    tol_: float = 1e-6, max_iter_: int = 30
):
    """Run a few EM steps to refresh GMM params for both classes."""
    if data_flat.device.type in {"mps", "cuda"}:
        return _em_once_sliced(
            data_flat[mask_fore_flat],
            data_flat[mask_back_flat],
            n_fore,
            n_back,
            pi,
            mu,
            sigma,
            tol_=tol_,
            max_iter_=max_iter_,
        )

    X_col = data_flat.reshape(-1, 1)
    mask_bg = mask_back_flat.reshape(-1, 1).to(dtype=data_flat.dtype)
    mask_fg = mask_fore_flat.reshape(-1, 1).to(dtype=data_flat.dtype)
    n_bg_total = mask_bg.sum().clamp_min(1.0)
    n_fg_total = mask_fg.sum().clamp_min(1.0)
    loglh_old = None
    for _ in range(max_iter_):
        resp_bg = _calculate_resp_masked(X_col, mask_bg, pi[:n_back, 0], mu[:n_back, 0], sigma[:n_back, 0])
        resp_fg = _calculate_resp_masked(X_col, mask_fg, pi[:n_fore, 1], mu[:n_fore, 1], sigma[:n_fore, 1])

        # M-step
        N_bg = resp_bg.sum(dim=0) + _TINY
        N_fg = resp_fg.sum(dim=0) + _TINY

        mu[:n_back, 0] = (1 / N_bg * (resp_bg * X_col)).sum(dim=0)
        mu[:n_fore, 1] = (1 / N_fg * (resp_fg * X_col)).sum(dim=0)
        sigma[:n_back, 0] = torch.sqrt((1 / N_bg * (resp_bg * (X_col - mu[:n_back, 0]) ** 2)).sum(dim=0)) + _TINY
        sigma[:n_fore, 1] = torch.sqrt((1 / N_fg * (resp_fg * (X_col - mu[:n_fore, 1]) ** 2)).sum(dim=0)) + _TINY

        pi[:n_back, 0] = N_bg / n_bg_total
        pi[:n_fore, 1] = N_fg / n_fg_total

        # Monitor inner EM convergence (optional)
        loglh_bg = _loglh_gmm_masked(X_col, mask_back_flat.to(dtype=data_flat.dtype), pi[:n_back, 0], mu[:n_back, 0], sigma[:n_back, 0])
        loglh_fg = _loglh_gmm_masked(X_col, mask_fore_flat.to(dtype=data_flat.dtype), pi[:n_fore, 1], mu[:n_fore, 1], sigma[:n_fore, 1])
        loglh_new = loglh_bg + loglh_fg
        if loglh_old is None:
            loglh_old = loglh_new
            continue
        if torch.abs((loglh_new - loglh_old) / (loglh_new + _TINY)) < tol_:
            break
        loglh_old = loglh_new

    return pi, mu, sigma


def _em_monitor_sample(data: torch.Tensor, max_points: int) -> torch.Tensor:
    """Return a deterministic, evenly spaced sample for convergence checks."""
    flat = data.reshape(-1)
    max_points = max(int(max_points), 1)
    if flat.numel() <= max_points:
        return flat
    stride = max((int(flat.numel()) + max_points - 1) // max_points, 1)
    return flat[::stride][:max_points]


def _outer_log_likelihood_monitor(
    label: torch.Tensor,
    class_likelihood: torch.Tensor,
    max_points: int = _EM_CONVERGENCE_MONITOR_POINTS,
    *,
    log_scores: bool = False,
) -> torch.Tensor:
    """Estimate outer log-likelihood on deterministic, evenly spaced voxels."""
    label_sample = _em_monitor_sample(label, max_points)
    background_sample = _em_monitor_sample(class_likelihood[0], max_points)
    foreground_sample = _em_monitor_sample(class_likelihood[1], max_points)
    selected = torch.where(label_sample == 0, background_sample, foreground_sample)
    if log_scores:
        return selected.sum()
    return torch.log(selected + _TINY).sum()


def _em_once_sliced_device_convergence(
    data_fore: torch.Tensor,
    data_back: torch.Tensor,
    n_fore: int,
    n_back: int,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    tol_: float = 1e-6,
    max_iter_: int = 30,
    monitor_points: int = _EM_CONVERGENCE_MONITOR_POINTS,
):
    """Run sliced EM while keeping convergence state on the accelerator."""
    data_back_col = data_back.reshape(-1, 1)
    data_fore_col = data_fore.reshape(-1, 1)
    monitor_back = _em_monitor_sample(data_back, monitor_points)
    monitor_fore = _em_monitor_sample(data_fore, monitor_points)
    active = torch.ones((), dtype=torch.bool, device=data_fore.device)
    loglh_old = torch.zeros((), dtype=data_fore.dtype, device=data_fore.device)

    for step in range(int(max_iter_)):
        resp_bg = _calculate_resp(data_back, pi[:n_back, 0], mu[:n_back, 0], sigma[:n_back, 0])
        resp_fg = _calculate_resp(data_fore, pi[:n_fore, 1], mu[:n_fore, 1], sigma[:n_fore, 1])

        N_bg = resp_bg.sum(dim=0) + _TINY
        N_fg = resp_fg.sum(dim=0) + _TINY
        candidate_pi = pi.clone()
        candidate_mu = mu.clone()
        candidate_sigma = sigma.clone()
        candidate_mu[:n_back, 0] = (1 / N_bg * (resp_bg * data_back_col)).sum(dim=0)
        candidate_mu[:n_fore, 1] = (1 / N_fg * (resp_fg * data_fore_col)).sum(dim=0)
        candidate_sigma[:n_back, 0] = torch.sqrt(
            (1 / N_bg * (resp_bg * (data_back_col - candidate_mu[:n_back, 0]) ** 2)).sum(dim=0)
        ) + _TINY
        candidate_sigma[:n_fore, 1] = torch.sqrt(
            (1 / N_fg * (resp_fg * (data_fore_col - candidate_mu[:n_fore, 1]) ** 2)).sum(dim=0)
        ) + _TINY
        candidate_pi[:n_back, 0] = N_bg / len(data_back)
        candidate_pi[:n_fore, 1] = N_fg / len(data_fore)

        # The converging update is retained, matching the host-side early-break
        # behavior. Once inactive, parameters remain frozen on the device.
        pi = torch.where(active, candidate_pi, pi)
        mu = torch.where(active, candidate_mu, mu)
        sigma = torch.where(active, candidate_sigma, sigma)

        loglh_new = _loglh_gmm(
            monitor_back,
            pi[:n_back, 0],
            mu[:n_back, 0],
            sigma[:n_back, 0],
        ) + _loglh_gmm(
            monitor_fore,
            pi[:n_fore, 1],
            mu[:n_fore, 1],
            sigma[:n_fore, 1],
        )
        if step > 0:
            delta = torch.abs((loglh_new - loglh_old) / (loglh_new + _TINY))
            active = active & (delta >= tol_)
        loglh_old = loglh_new

    return pi, mu, sigma


def _em_once_sliced(
    data_fore: torch.Tensor,
    data_back: torch.Tensor,
    n_fore: int,
    n_back: int,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    tol_: float = 1e-6,
    max_iter_: int = 30,
):
    """Original sliced EM path; faster for single large 3D volumes."""
    if data_fore.device.type in {"mps", "cuda"}:
        return _em_once_sliced_device_convergence(
            data_fore,
            data_back,
            n_fore,
            n_back,
            pi,
            mu,
            sigma,
            tol_=tol_,
            max_iter_=max_iter_,
        )

    loglh_old = None
    data_back_col = data_back.reshape(-1, 1)
    data_fore_col = data_fore.reshape(-1, 1)
    for _ in range(max_iter_):
        resp_bg = _calculate_resp(data_back, pi[:n_back, 0], mu[:n_back, 0], sigma[:n_back, 0])
        resp_fg = _calculate_resp(data_fore, pi[:n_fore, 1], mu[:n_fore, 1], sigma[:n_fore, 1])

        N_bg = resp_bg.sum(dim=0) + _TINY
        N_fg = resp_fg.sum(dim=0) + _TINY

        mu[:n_back, 0] = (1 / N_bg * (resp_bg * data_back_col)).sum(dim=0)
        mu[:n_fore, 1] = (1 / N_fg * (resp_fg * data_fore_col)).sum(dim=0)
        sigma[:n_back, 0] = torch.sqrt((1 / N_bg * (resp_bg * (data_back_col - mu[:n_back, 0]) ** 2)).sum(dim=0)) + _TINY
        sigma[:n_fore, 1] = torch.sqrt((1 / N_fg * (resp_fg * (data_fore_col - mu[:n_fore, 1]) ** 2)).sum(dim=0)) + _TINY

        pi[:n_back, 0] = N_bg / len(data_back)
        pi[:n_fore, 1] = N_fg / len(data_fore)

        loglh_bg = _loglh_gmm(data_back, pi[:n_back, 0], mu[:n_back, 0], sigma[:n_back, 0])
        loglh_fg = _loglh_gmm(data_fore, pi[:n_fore, 1], mu[:n_fore, 1], sigma[:n_fore, 1])
        loglh_new = loglh_bg + loglh_fg
        if loglh_old is None:
            loglh_old = loglh_new
            continue
        if torch.abs((loglh_new - loglh_old) / (loglh_new + _TINY)) < tol_:
            break
        loglh_old = loglh_new

    return pi, mu, sigma


def _sample_1d_tensor(values: torch.Tensor, count: int) -> torch.Tensor:
    """Uniformly sample a 1D tensor with replacement.

    Sampling with replacement avoids creating a full ``randperm`` for very large
    volumes, which is expensive on GPU/MPS. For EM/GMM fitting this is an
    unbiased stochastic approximation of the full class distribution.
    """
    values = values.reshape(-1)
    count = int(count)
    if count <= 0 or values.numel() <= count:
        return values
    idx = torch.randint(values.numel(), (count,), device=values.device)
    return values[idx]


def _sample_masked_1d_tensor(
    values: torch.Tensor,
    mask: torch.Tensor,
    count: int | None,
    *,
    valid_count: int | None = None,
    chunk_size: int = 1_048_576,
) -> torch.Tensor:
    """Uniformly sample masked values without materializing the full class."""
    values = values.reshape(-1)
    mask = mask.reshape(-1).to(dtype=torch.bool)
    valid_count = int(mask.sum().item()) if valid_count is None else int(valid_count)
    requested = int(count or 0)
    if valid_count < 1:
        return values[:0]
    if requested <= 0 or valid_count <= requested:
        return values[mask]

    requested = min(requested, valid_count)
    sampled = torch.empty(requested, dtype=values.dtype, device=values.device)
    acceptance = max(valid_count / max(values.numel(), 1), 1.0 / max(values.numel(), 1))
    written = 0
    # Device-side rejection sampling avoids the per-chunk `.item()` syncs that
    # are particularly expensive on MPS. Conditional on mask=True, accepted
    # flat indices are uniform and sampling remains with replacement.
    max_batch = max(4_000_000, int(chunk_size), requested * 2)
    for _attempt in range(64):
        remaining = requested - written
        if remaining <= 0:
            break
        draw_count = int(np.ceil(remaining / acceptance * 1.2)) + 1024
        draw_count = min(max(draw_count, remaining), max_batch)
        candidates = torch.randint(
            values.numel(),
            (draw_count,),
            device=values.device,
        )
        accepted_indices = candidates[mask[candidates]]
        accepted_count = min(int(accepted_indices.numel()), remaining)
        if accepted_count > 0:
            sampled[written : written + accepted_count] = values[accepted_indices[:accepted_count]]
            written += accepted_count

    if written != requested:
        # Extremely sparse masks can defeat bounded rejection batches. Fall
        # back only in that case so correctness is never traded for speed.
        return _sample_1d_tensor(values[mask], requested)
    return sampled


def _sample_gmm_training_data_from_masks(
    data_flat: torch.Tensor,
    mask_fg_flat: torch.Tensor,
    mask_bg_flat: torch.Tensor,
    max_points: int | None,
    background_max_points: int | None = None,
    *,
    foreground_max_points: int | None = None,
    valid_counts: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather only the foreground/background points that EM will consume."""
    if valid_counts is None:
        counts = torch.stack((mask_fg_flat.sum(), mask_bg_flat.sum())).detach().cpu()
        n_fg, n_bg = (int(value) for value in counts.tolist())
    else:
        n_fg, n_bg = (int(value) for value in valid_counts)
    fg_count = 0
    bg_count = 0

    if max_points is not None and int(max_points) > 0:
        max_points = int(max_points)
        total = n_fg + n_bg
        if total > max_points and n_fg > 0 and n_bg > 0:
            fg_count = min(max(1, int(round(max_points * (n_fg / total)))), n_fg)
            bg_count = min(max(1, max_points - fg_count), n_bg)
            spare = max_points - fg_count - bg_count
            if spare > 0 and fg_count < n_fg:
                add = min(spare, n_fg - fg_count)
                fg_count += add
                spare -= add
            if spare > 0 and bg_count < n_bg:
                bg_count += min(spare, n_bg - bg_count)
    else:
        if foreground_max_points is not None and int(foreground_max_points) > 0:
            fg_count = min(int(foreground_max_points), n_fg)
        if background_max_points is not None and int(background_max_points) > 0:
            bg_count = min(int(background_max_points), n_bg)

    data_fg = _sample_masked_1d_tensor(
        data_flat,
        mask_fg_flat,
        fg_count,
        valid_count=n_fg,
    )
    data_bg = _sample_masked_1d_tensor(
        data_flat,
        mask_bg_flat,
        bg_count,
        valid_count=n_bg,
    )
    return data_fg, data_bg


# ---------------------
# Public API
# ---------------------

def segmentation(
    image: np.ndarray,
    frangi: np.ndarray,
    pixel_size: tuple[float, float] | tuple[float, float, float],
    beta1: float,
    beta2: float,
    *,
    n_fore: int = 3,
    n_back: int = 8,
    max_iter: int = 50,
    device: str | torch.device = "cpu",
    progress: Callable[[int, float], None] | None = None,
    tol: float = _DEFAULT_TOL,
    initial_state: dict[str, np.ndarray] | None = None,
    init_method: str = "random",
    em_sample_points: int | None = None,
    em_foreground_sample_points: int | None = _DEFAULT_EM_CLASS_SAMPLE_POINTS,
    em_background_sample_points: int | None = _DEFAULT_EM_CLASS_SAMPLE_POINTS,
    random_seed: int | None = 0,
) -> tuple[np.ndarray, SegmentationInfo]:
    """
    Graphical model segmentation with GMM unary + pairwise smoothness.
    ``beta1`` weights pairwise spatial smoothness and ``beta2`` weights the
    structural-response unary potential, matching the notation in the paper.
    Returns (label_uint8, SegmentationInfo). If `progress` is provided, it is
    called every outer iteration with (iteration_index, delta_loglh).

    Parameters
    ----------
    image : np.ndarray
        2D/3D array; we expect C-first after pre-processing (Frangi already matched).
    frangi : np.ndarray
        Same shape as image; used for frangi potentials.
    pixel_size : old-plugin compatible interpretation.
        2D uses pixel_size[0] as xy spacing.
        3D uses pixel_size[0] as z spacing and pixel_size[1] as xy spacing.
    device : str | torch.device
        "cpu", "cuda", "mps", etc.
    n_fore : int
        Number of foreground GMM components. Default is 3.
    n_back : int
        Number of background GMM components. Default is 8.
    em_sample_points : int | None
        If positive, cap the total foreground+background points used for GMM
        initialization and EM updates. Label updates still run on the full image.
    em_foreground_sample_points : int | None
        If positive and `em_sample_points` is not positive, sample at most this
        many foreground points for GMM initialization and EM updates.
    em_background_sample_points : int | None
        If positive and `em_sample_points` is not positive, sample at most this
        many background points for GMM initialization and EM updates.
    """
    image = np.asarray(image)
    frangi = np.asarray(frangi)
    if image.shape != frangi.shape:
        raise ValueError(
            f"image and frangi must have the same shape, got {image.shape} and {frangi.shape}."
        )
    if image.ndim not in {2, 3}:
        raise ValueError(f"segmentation expects a 2D or 3D image, got ndim={image.ndim}.")
    if int(n_fore) < 1 or int(n_back) < 1:
        raise ValueError("n_fore and n_back must both be at least 1.")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be at least 1.")
    required_spacings = 1 if image.ndim == 2 else 2
    if len(pixel_size) < required_spacings:
        raise ValueError(
            f"pixel_size needs at least {required_spacings} values for {image.ndim}D segmentation."
        )
    if any(not np.isfinite(float(value)) or float(value) <= 0 for value in pixel_size):
        raise ValueError("pixel_size values must be finite and positive.")
    if not all(np.isfinite(float(value)) for value in (beta1, beta2, tol)):
        raise ValueError("beta1, beta2, and tol must be finite.")
    init_method = str(init_method).lower()
    if init_method not in {"random", "otsu"}:
        raise ValueError("init_method must be 'random' or 'otsu'.")

    with torch.inference_mode():
        dev = torch.device(device)
        if random_seed is not None:
            seed = int(random_seed)
            torch.manual_seed(seed)
            if dev.type == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            elif dev.type == "mps" and hasattr(torch.mps, "manual_seed"):
                torch.mps.manual_seed(seed)
        data = _as_tensor_without_readonly_alias(image, device=dev).unsqueeze(0)
        frg = _as_tensor_without_readonly_alias(frangi, device=dev).unsqueeze(0)

        # normalize image to [0, 255] like the original
        data = (data - data.min()) / (data.max() - data.min() + _TINY) * 255.0

        n_component = max(n_fore, n_back)
        class_num = 2
        fpot0, fpot1 = _frangi_potential(frg, float(beta2))

        # Shape branches
        if data.ndim == 3:
            # (C,H,W)
            C, H, W = data.shape
            label = torch.randint(low=0, high=2, size=(C, H, W), device=dev, dtype=torch.uint8)
            pixel_size_xy = float(pixel_size[0])
            pixel_size_z = None
            U_g = torch.zeros((class_num, H, W), device=dev)
        else:
            # (C,D,H,W)
            C, D, H, W = data.shape
            label = torch.randint(low=0, high=2, size=(C, D, H, W), device=dev, dtype=torch.uint8)
            pixel_size_z = float(pixel_size[0])
            pixel_size_xy = float(pixel_size[1])
            U_g = torch.zeros((class_num, D, H, W), device=dev)
        U_c = torch.empty_like(U_g)

        # GMM params
        pi = torch.zeros((n_component, class_num), device=dev)
        mu = torch.zeros((n_component, class_num), device=dev)
        sigma = torch.zeros((n_component, class_num), device=dev)
        warm_started = False

        if initial_state is not None:
            try:
                init_label = _as_tensor_without_readonly_alias(
                    initial_state["label"], device=dev
                )
                init_pi = _as_tensor_without_readonly_alias(
                    initial_state["pi"], device=dev
                )
                init_mu = _as_tensor_without_readonly_alias(
                    initial_state["mu"], device=dev
                )
                init_sigma = _as_tensor_without_readonly_alias(
                    initial_state["sigma"], device=dev
                )
                if (
                    tuple(init_label.shape) == tuple(label.shape)
                    and tuple(init_pi.shape) == tuple(pi.shape)
                    and tuple(init_mu.shape) == tuple(mu.shape)
                    and tuple(init_sigma.shape) == tuple(sigma.shape)
                ):
                    label.copy_(init_label.to(dtype=label.dtype))
                    pi.copy_(init_pi.to(dtype=pi.dtype))
                    mu.copy_(init_mu.to(dtype=mu.dtype))
                    sigma.copy_(init_sigma.to(dtype=sigma.dtype))
                    warm_started = bool(
                        torch.isfinite(pi).all()
                        and torch.isfinite(mu).all()
                        and torch.isfinite(sigma).all()
                    )
            except (KeyError, TypeError, ValueError):
                warm_started = False

        deltas: list[float] = []
        loglh_old_outer: torch.Tensor | None = None
        data_flat = data.reshape(-1)

        for it in range(int(max_iter)):
            if it == 0 and not warm_started and init_method == "otsu":
                thr = _threshold_otsu_torch(data)
                label = (data > thr).to(dtype=label.dtype)

            label_flat = label.reshape(-1)
            mask_bg_flat = label_flat == 0
            mask_fg_flat = label_flat == 1

            # Read both counts with one device synchronization, then reuse them
            # for validation, sampling, and any component swap this iteration.
            class_counts = torch.stack((mask_fg_flat.sum(), mask_bg_flat.sum())).detach().cpu()
            n_fg_count, n_bg_count = (int(value) for value in class_counts.tolist())

            if n_bg_count < 1 or n_fg_count < 1:
                # Keep mask shape compatible with `label` (with the channel dim).
                mask = data > data.median()
                label[mask] = 1
                label_flat = label.reshape(-1)
                mask_bg_flat = label_flat == 0
                mask_fg_flat = label_flat == 1
                class_counts = torch.stack((mask_fg_flat.sum(), mask_bg_flat.sum())).detach().cpu()
                n_fg_count, n_bg_count = (int(value) for value in class_counts.tolist())

            if it == 0 and not warm_started:
                init_fg, init_bg = _sample_gmm_training_data_from_masks(
                    data_flat,
                    mask_fg_flat,
                    mask_bg_flat,
                    em_sample_points,
                    em_background_sample_points,
                    foreground_max_points=em_foreground_sample_points,
                    valid_counts=(n_fg_count, n_bg_count),
                )
                pi, mu, sigma = _parameter_initialization(
                    dev,
                    pi,
                    mu,
                    sigma,
                    init_fg,
                    init_bg,
                    n_fore,
                    n_back,
                    random_state=random_seed,
                )

            if data.ndim == 3:
                if (
                    (em_sample_points is not None and int(em_sample_points) > 0)
                    or (
                        em_foreground_sample_points is not None
                        and int(em_foreground_sample_points) > 0
                    )
                    or (
                        em_background_sample_points is not None
                        and int(em_background_sample_points) > 0
                    )
                ):
                    em_fg, em_bg = _sample_gmm_training_data_from_masks(
                        data_flat,
                        mask_fg_flat,
                        mask_bg_flat,
                        em_sample_points,
                        em_background_sample_points,
                        foreground_max_points=em_foreground_sample_points,
                        valid_counts=(n_fg_count, n_bg_count),
                    )
                    pi, mu, sigma = _em_once_sliced(
                        em_fg,
                        em_bg,
                        n_fore,
                        n_back,
                        pi,
                        mu,
                        sigma,
                        tol_=1e-6,
                        max_iter_=30,
                    )
                else:
                    pi, mu, sigma = _em_once(
                        data_flat,
                        mask_fg_flat,
                        mask_bg_flat,
                        n_fore,
                        n_back,
                        pi,
                        mu,
                        sigma,
                        tol_=1e-6,
                        max_iter_=30,
                    )
            else:
                data_fg, data_bg = _sample_gmm_training_data_from_masks(
                    data_flat,
                    mask_fg_flat,
                    mask_bg_flat,
                    em_sample_points,
                    em_background_sample_points,
                    foreground_max_points=em_foreground_sample_points,
                    valid_counts=(n_fg_count, n_bg_count),
                )
                pi, mu, sigma = _em_once_sliced(
                    data_fg,
                    data_bg,
                    n_fore,
                    n_back,
                    pi,
                    mu,
                    sigma,
                    tol_=1e-6,
                    max_iter_=30,
                )

            if dev.type in {"mps", "cuda"}:
                pi, mu, sigma = _switch_parameters_device(
                    pi,
                    mu,
                    sigma,
                    n_fore,
                    n_back,
                    n_fg_count,
                    n_bg_count,
                )
            elif torch.min(mu[:n_fore, 1]) < torch.max(mu[:n_back, 0]):
                pi, mu, sigma = _switch_parameters(
                    pi,
                    mu,
                    sigma,
                    label,
                    n_fore,
                    n_back,
                )

            if data.ndim == 3:
                _pairwise_potential(
                    label,
                    float(beta1),
                    0.1,
                    fpot0,
                    fpot1,
                    dev,
                    pixel_size_xy,
                    output=U_c,
                )
                _fill_gmm_class_likelihood(
                    U_g[0],
                    data,
                    pi[:n_back, 0],
                    mu[:n_back, 0],
                    sigma[:n_back, 0],
                    U_c[0],
                    log_output=True,
                )
                _fill_gmm_class_likelihood(
                    U_g[1],
                    data,
                    pi[:n_fore, 1],
                    mu[:n_fore, 1],
                    sigma[:n_fore, 1],
                    U_c[1],
                    log_output=True,
                )
                label = _binary_label_from_likelihood(U_g, (C, H, W))
            else:
                _pairwise_potential(
                    label,
                    float(beta1),
                    0.1,
                    fpot0,
                    fpot1,
                    dev,
                    pixel_size_xy,
                    pixel_size_z,
                    output=U_c,
                )
                _fill_gmm_class_likelihood(
                    U_g[0],
                    data,
                    pi[:n_back, 0],
                    mu[:n_back, 0],
                    sigma[:n_back, 0],
                    U_c[0],
                    log_output=True,
                )
                _fill_gmm_class_likelihood(
                    U_g[1],
                    data,
                    pi[:n_fore, 1],
                    mu[:n_fore, 1],
                    sigma[:n_fore, 1],
                    U_c[1],
                    log_output=True,
                )
                label = _binary_label_from_likelihood(U_g, (C, D, H, W))

            loglh_new_outer = _outer_log_likelihood_monitor(
                label,
                U_g,
                log_scores=True,
            )

            if loglh_old_outer is not None:
                delta = torch.abs((loglh_new_outer - loglh_old_outer) / (loglh_new_outer + _TINY))
                # Read the scalar once.  Each MPS/CUDA -> CPU conversion is a
                # synchronization point, so repeating it makes every outer
                # iteration pause more than necessary.
                delta_value = float(delta.detach().cpu())
                deltas.append(delta_value)
                if progress is not None:
                    progress(it, delta_value)
                if delta_value < tol:
                    info = SegmentationInfo(
                        iterations_run=it + 1,
                        deltas=deltas,
                        converged=True,
                        state={
                            "label": label.detach().cpu().numpy().copy(),
                            "pi": pi.detach().cpu().numpy().copy(),
                            "mu": mu.detach().cpu().numpy().copy(),
                            "sigma": sigma.detach().cpu().numpy().copy(),
                        },
                    )
                    out = (label * 255).detach().cpu().numpy().astype(np.uint8)
                    return out, info
            else:
                if progress is not None:
                    progress(it, float("nan"))

            loglh_old_outer = loglh_new_outer

        info = SegmentationInfo(
            iterations_run=int(max_iter),
            deltas=deltas,
            converged=False,
            state={
                "label": label.detach().cpu().numpy().copy(),
                "pi": pi.detach().cpu().numpy().copy(),
                "mu": mu.detach().cpu().numpy().copy(),
                "sigma": sigma.detach().cpu().numpy().copy(),
            },
        )
        out = (label * 255).detach().cpu().numpy().astype(np.uint8)
        return out, info
