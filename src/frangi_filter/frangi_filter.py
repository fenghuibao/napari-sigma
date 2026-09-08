#!/usr/bin/env python3
# Author: Huibao Feng
# Date: 2025-11-02

import numpy as np
import torch
from torch import nn

from .gaussian_smoothing import GaussianSmoothing


_FRANGI_RESPONSE_CHUNK_VOXELS = 2_097_152
_CUDA_AUTO_MAX_SIGMA_STREAMS = 2
_CUDA_PARALLEL_MEMORY_SAFETY = 1.5


def symmetric_eigvalsh_3x3(elements):
    """Return descending eigenvalues of symmetric 3x3 matrices.

    ``elements`` must contain ``(M00, M01, M02, M11, M12, M22)`` tensors.
    The closed-form solution uses only elementwise torch operations, keeping
    the calculation on the input device (CPU, CUDA, or MPS) without creating
    an ``(..., 3, 3)`` matrix tensor.
    """
    if len(elements) != 6:
        raise ValueError("A symmetric 3x3 matrix requires six elements")
    m00, m01, m02, m11, m12, m22 = elements
    if not all(value.shape == m00.shape for value in elements[1:]):
        raise ValueError("All symmetric matrix elements must have the same shape")
    if not all(value.device == m00.device for value in elements[1:]):
        raise ValueError("All symmetric matrix elements must be on the same device")
    if not all(value.dtype == m00.dtype for value in elements[1:]):
        raise ValueError("All symmetric matrix elements must have the same dtype")
    if not m00.is_floating_point():
        raise TypeError("Symmetric eigendecomposition requires floating-point tensors")

    # Normalize each matrix independently. This prevents p**3 in the cubic
    # solution from overflowing or underflowing for very large/small Hessians.
    scale = m00.abs()
    for value in (m01, m02, m11, m12, m22):
        scale = torch.maximum(scale, value.abs())
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))

    a = m00 / safe_scale
    b = m01 / safe_scale
    c = m02 / safe_scale
    d = m11 / safe_scale
    e = m12 / safe_scale
    f = m22 / safe_scale

    q = (a + d + f) / 3.0
    a0 = a - q
    d0 = d - q
    f0 = f - q
    p_squared = (
        a0.square()
        + d0.square()
        + f0.square()
        + 2.0 * (b.square() + c.square() + e.square())
    ) / 6.0
    p = torch.sqrt(torch.clamp_min(p_squared, 0.0))

    # For a scalar matrix p is zero and all eigenvalues equal q. Use a safe
    # denominator in the unused analytic branch to avoid NaNs during eager
    # evaluation, then select the repeated root below.
    repeated = p <= (torch.finfo(m00.dtype).eps * 4.0)
    safe_p = torch.where(repeated, torch.ones_like(p), p)
    det_centered = (
        a0 * d0 * f0
        + 2.0 * b * c * e
        - a0 * e.square()
        - d0 * c.square()
        - f0 * b.square()
    )
    r = det_centered / (2.0 * safe_p.pow(3))
    phi = torch.acos(torch.clamp(r, -1.0, 1.0)) / 3.0

    largest = q + 2.0 * p * torch.cos(phi)
    smallest = q + 2.0 * p * torch.cos(phi + (2.0 * np.pi / 3.0))
    middle = 3.0 * q - largest - smallest
    largest = torch.where(repeated, q, largest)
    middle = torch.where(repeated, q, middle)
    smallest = torch.where(repeated, q, smallest)

    return torch.stack((largest, middle, smallest), dim=0) * scale.unsqueeze(0)


class FrangiFilter(nn.Module):

    def __init__(
        self,
        channels,
        kernel_size,
        sigmas,
        dim,
        device='cpu',
        zx_ratio=1,
        psf_ratio=3.0,
        alpha=0.5,
        beta=0.5,
        gamma=2,
        response_mode="vesselness",
        sigma_parallelism="auto",
    ):
        """
        Arguments:
            channels (int, sequence): Number of channels of the input tensors. Output will
                have this number of channels as well.
            kernel_size (int, sequence): Size of the gaussian kernel.
            sigmas (list, sequence): List of standard deviations of the gaussian kernels.
                Interpreted in xy-pixel units; the corresponding z-pixel sigma is
                ``sigma * psf_ratio / zx_ratio`` so the filter matches the PSF-broadened
                apparent feature size (see ``psf_ratio`` below).
            zx_ratio (float): ``pixel_size_z / pixel_size_xy`` of the input volume.
            psf_ratio (float): ``PSF_z / PSF_xy`` of the optical system. For confocal
                microscopy ``psf_ratio ≈ 3`` is a good default. Set to 1.0 if the
                volume is already isotropic (e.g. after deconvolution or for SIM
                reconstructions); the formula then collapses to the physically
                isotropic ``sigma / zx_ratio``.
            alpha (float): Vessel/sheet shape parameter.
            beta (float): Blobness suppression parameter.
            gamma (float): Structuredness parameter.
            response_mode (str): "vesselness", "sheetness", or "combined".
                Default value is 2 (spatial).
            sigma_parallelism ("auto", "serial", or int): CUDA sigma stream
                count. Auto measures the first scale's actual peak memory and
                enables two streams only when the current device has room.
                CPU and MPS remain serial because their operators already use
                internal parallelism and MPS has no safe public stream API.
        """
        super().__init__()
        self.channels = channels
        self.sigmas = sigmas
        self.kernel_size= kernel_size
        self.dim = dim
        self.device = device
        self.alpha = max(float(alpha), 1e-12)
        self.beta = max(float(beta), 1e-12)
        self.gamma = max(float(gamma), 1e-12)
        self.response_mode = str(response_mode or "vesselness").strip().lower()
        if self.response_mode not in {"vesselness", "sheetness", "combined"}:
            raise ValueError(
                "response_mode must be one of: vesselness, sheetness, combined"
            )
        self.sigma_parallelism = sigma_parallelism
        self.zx_ratio = zx_ratio
        self.psf_ratio = float(psf_ratio)
        self._filters = nn.ModuleDict()
        self.register_buffer(
            "_eps",
            torch.tensor(1e-15, dtype=torch.float32, device=self.device),
        )

        # Cache derivative filters per sigma to avoid rebuilding kernels for each run.
        # The first derivative filters are applied twice to form each Hessian
        # element. Using sigma/sqrt(2) here makes the effective second-derivative
        # scale equal to `sigma`.
        coef = 1 / np.sqrt(2)
        for sigma in self.sigmas:
            sigma_key = self._sigma_key(sigma)
            if self.dim == 2:
                self._filters[f"{sigma_key}:x"] = GaussianSmoothing(
                    channels=self.channels,
                    kernel_size=self.kernel_size,
                    sigma=coef * sigma,
                    dim=2,
                    order="x",
                    device=self.device,
                )
                self._filters[f"{sigma_key}:y"] = GaussianSmoothing(
                    channels=self.channels,
                    kernel_size=self.kernel_size,
                    sigma=coef * sigma,
                    dim=2,
                    order="y",
                    device=self.device,
                )
            else:
                xy_sigma = coef * float(sigma)
                z_sigma = xy_sigma * self._psf_sampling_factor()
                sigma_zyx = (z_sigma, xy_sigma, xy_sigma)
                self._filters[f"{sigma_key}:x"] = GaussianSmoothing(
                    channels=self.channels,
                    kernel_size=self.kernel_size,
                    sigma=sigma_zyx,
                    dim=3,
                    order="x",
                    device=self.device,
                )
                self._filters[f"{sigma_key}:y"] = GaussianSmoothing(
                    channels=self.channels,
                    kernel_size=self.kernel_size,
                    sigma=sigma_zyx,
                    dim=3,
                    order="y",
                    device=self.device,
                )
                # σ_z pixel = σ_xy_px × PSF_z/PSF_xy ÷ (Δz/Δxy).
                # · zx_ratio compensates for anisotropic *sampling*  (pixel spacing)
                # · psf_ratio compensates for anisotropic *imaging*  (PSF stretching)
                # With confocal defaults (psf_ratio=3, zx_ratio≈1.5) this gives
                # σ_z ≈ 2 × σ_xy, which matches the apparent z extent of a
                # sub-PSF feature. psf_ratio=1 collapses to σ_z = σ_xy / zx_ratio
                # for already-isotropic data.
                self._filters[f"{sigma_key}:z"] = GaussianSmoothing(
                    channels=self.channels,
                    kernel_size=self.kernel_size,
                    sigma=sigma_zyx,
                    dim=3,
                    order="z",
                    device=self.device,
                )

    @staticmethod
    def _sigma_key(sigma):
        return f"{float(sigma):.8f}".replace(".", "_")

    def _get_filter(self, sigma, axis):
        return self._filters[f"{self._sigma_key(sigma)}:{axis}"]

    def _psf_sampling_factor(self) -> float:
        if self.zx_ratio > 0:
            return float(self.psf_ratio) / float(self.zx_ratio)
        return float(self.psf_ratio)

    def _effective_sigma_zyx(self, sigma: float) -> tuple[float, float, float]:
        sigma_xy = float(sigma)
        sigma_z = sigma_xy * self._psf_sampling_factor()
        return sigma_z, sigma_xy, sigma_xy

    def _scale_normalize_hessian(self, hessian_elems, sigma: float):
        if self.dim == 2:
            scale = float(sigma) ** 2
            return tuple(elem.mul_(scale) for elem in hessian_elems)

        sigma_z, sigma_y, sigma_x = self._effective_sigma_zyx(sigma)
        Hzz, Hzy, Hzx, Hyy, Hyx, Hxx = hessian_elems
        return (
            Hzz.mul_(sigma_z * sigma_z),
            Hzy.mul_(sigma_z * sigma_y),
            Hzx.mul_(sigma_z * sigma_x),
            Hyy.mul_(sigma_y * sigma_y),
            Hyx.mul_(sigma_y * sigma_x),
            Hxx.mul_(sigma_x * sigma_x),
        )

    @staticmethod
    def _sort3_by_abs(eigvals):
        """Sort three eigenvalue images by absolute value without torch.argsort."""
        a, b, c = eigvals[0], eigvals[1], eigvals[2]
        swap = a.abs() > b.abs()
        a, b = torch.where(swap, b, a), torch.where(swap, a, b)
        swap = b.abs() > c.abs()
        b, c = torch.where(swap, c, b), torch.where(swap, b, c)
        swap = a.abs() > b.abs()
        a, b = torch.where(swap, b, a), torch.where(swap, a, b)
        return torch.stack((a, b, c), dim=0)


    def Hessian_matrix(self, image, sigma):
        if self.dim == 2:
            Gx = self._get_filter(sigma, 'x')
            Gy = self._get_filter(sigma, 'y')
            Iy = Gy(image)
            Hxy = Gx(Iy)
            Hyy = Gy(Iy)
            del Iy
            Ix = Gx(image)
            Hxx = Gx(Ix)
            del Ix
            return Hyy, Hxy, Hxx
        else:
            Gx = self._get_filter(sigma, 'x')
            Gy = self._get_filter(sigma, 'y')
            Gz = self._get_filter(sigma, 'z')
            Ix = Gx(image)
            Hzx = Gz(Ix)
            Hyx = Gy(Ix)
            Hxx = Gx(Ix)
            del Ix
            Iy = Gy(image)
            Hzy = Gz(Iy)
            Hyy = Gy(Iy)
            del Iy
            Iz = Gz(image)
            Hzz = Gz(Iz)
            del Iz
            return Hzz, Hzy, Hzx, Hyy, Hyx, Hxx


    def _symmetric_compute_eigenvalues(self, S_elems):
        if len(S_elems) == 3:
            M00, M01, M11 = S_elems
            mean = (M00 + M11) * 0.5
            hsqrtdet = torch.hypot(M01, (M00 - M11) * 0.5)  # sqrt(x^2 + y^2) elementwise
            eigs = torch.stack([mean + hsqrtdet, mean - hsqrtdet], dim=0)
        else:
            eigs = symmetric_eigvalsh_3x3(S_elems)
        return torch.squeeze(eigs)

    def _response_from_eigenvalues(self, eigvals):
        eps = self._eps
        if self.dim == 2:
            abs0 = eigvals[0].abs()
            abs1 = eigvals[1].abs()
            swap = abs0 > abs1
            lambda1 = torch.where(swap, eigvals[1], eigvals[0])
            lambda2_signed = torch.where(swap, eigvals[0], eigvals[1])
            lambda2 = torch.maximum(lambda2_signed, eps)
            r_b = lambda1.abs() / lambda2
            # The caller negates the input so that bright tubes in the
            # original image become dark valleys here. For a dark valley
            # the cross-tube curvature is positive, i.e. lambda2 > 0.
            sign_mask = lambda2_signed > 0
        else:  # ndim == 3
            eigvals = self._sort3_by_abs(eigvals)
            lambda1 = eigvals[0]
            lambda2_signed = eigvals[1]
            lambda3_signed = eigvals[2]
            lambda2 = torch.maximum(lambda2_signed, eps)
            lambda3 = torch.maximum(lambda3_signed, eps)
            r_a = lambda2 / lambda3
            r_b = lambda1.abs() / torch.sqrt(lambda2 * lambda3)
            # Same idea as the 2D branch: a bright tube/plate/blob in the
            # original image (=> dark in the negated input) needs both
            # cross-axis curvatures positive after sorting.
            sign_mask = (lambda2_signed > 0) & (lambda3_signed > 0)

        s = torch.sqrt((eigvals**2).sum(dim=0))
        structuredness = 1.0 - torch.exp(-(s**2) / (2 * (self.gamma**2)))
        if self.dim == 2:
            vesselness = torch.exp(-(r_b**2) / (2 * (self.beta**2)))
        else:
            vesselness = 1.0 - torch.exp(-(r_a**2) / (2 * (self.alpha**2)))
            vesselness = vesselness * torch.exp(-(r_b**2) / (2 * (self.beta**2)))
        vesselness = vesselness * structuredness
        vesselness = torch.where(sign_mask, vesselness, torch.zeros_like(vesselness))

        if self.dim == 3:
            # A sheet/disk has one strong curvature direction and two weak
            # tangential directions: |lambda1| ~= |lambda2| ~= 0 << |lambda3|.
            lambda3_abs = torch.maximum(lambda3_signed.abs(), eps)
            sheet_ra = lambda2_signed.abs() / lambda3_abs
            sheet_rb = lambda1.abs() / lambda3_abs
            sheetness = torch.exp(-(sheet_ra**2) / (2 * (self.alpha**2)))
            sheetness = sheetness * torch.exp(-(sheet_rb**2) / (2 * (self.beta**2)))
            sheetness = sheetness * structuredness
            sheetness = torch.where(lambda3_signed > 0, sheetness, torch.zeros_like(sheetness))
        else:
            # Sheetness is not well-defined for a single 2D Hessian plane.
            sheetness = vesselness

        if self.response_mode == "sheetness":
            vals = sheetness
        elif self.response_mode == "combined":
            vals = torch.maximum(vesselness, sheetness)
        else:
            vals = vesselness
        return torch.nan_to_num(vals, nan=0.0)

    def _calc_sigma_response(self, image, sigma):
        hessian_elems = self._scale_normalize_hessian(
            self.Hessian_matrix(image, sigma),
            sigma,
        )
        # The analytic eigensolver has many per-voxel intermediates. Running it
        # over the whole volume at once multiplies peak memory without changing
        # the independent voxel calculations, so bound its working set.
        # The flattened destination must share storage even for transposed input.
        response = torch.empty_like(image, memory_format=torch.contiguous_format)
        response_flat = response.view(-1)
        hessian_flat = tuple(element.reshape(-1) for element in hessian_elems)
        for start in range(0, response_flat.numel(), _FRANGI_RESPONSE_CHUNK_VOXELS):
            stop = min(start + _FRANGI_RESPONSE_CHUNK_VOXELS, response_flat.numel())
            eigvals = self._symmetric_compute_eigenvalues(
                tuple(element[start:stop] for element in hessian_flat)
            )
            response_flat[start:stop] = self._response_from_eigenvalues(eigvals)
        return response

    def _requested_cuda_sigma_streams(self, pending: int) -> int:
        pending = max(int(pending), 0)
        if pending < 2:
            return 1
        value = self.sigma_parallelism
        if isinstance(value, str):
            mode = value.strip().lower()
            if mode in {"serial", "off", "false", "1"}:
                return 1
            if mode != "auto":
                raise ValueError("sigma_parallelism must be 'auto', 'serial', or a positive integer")
            return min(_CUDA_AUTO_MAX_SIGMA_STREAMS, pending)
        return min(max(int(value), 1), pending)

    @staticmethod
    def _memory_limited_sigma_streams(
        requested: int,
        free_bytes: int,
        measured_peak_bytes: int,
    ) -> int:
        requested = max(int(requested), 1)
        measured_peak_bytes = max(int(measured_peak_bytes), 1)
        bytes_per_stream = measured_peak_bytes * _CUDA_PARALLEL_MEMORY_SAFETY
        memory_limit = max(int(float(free_bytes) // bytes_per_stream), 1)
        return min(requested, memory_limit)

    @staticmethod
    def _sync_progress_device(image) -> None:
        if image.device.type == "mps":
            torch.mps.synchronize()
        elif image.device.type == "cuda":
            torch.cuda.current_stream(image.device).synchronize()

    def _merge_sigma_response(self, filtered_max, vals) -> None:
        torch.maximum(filtered_max, vals, out=filtered_max)

    def _calc_frangi_response(self, image, progress_callback=None):
        filtered_max = torch.zeros_like(image)
        total = len(self.sigmas)
        if total < 1:
            return filtered_max

        cuda_parallel_requested = (
            image.device.type == "cuda"
            and total > 2
            and self._requested_cuda_sigma_streams(total) > 1
        )
        cuda_baseline = 0
        if cuda_parallel_requested:
            torch.cuda.current_stream(image.device).synchronize()
            cuda_baseline = int(torch.cuda.memory_allocated(image.device))
            torch.cuda.reset_peak_memory_stats(image.device)

        first = self._calc_sigma_response(image, self.sigmas[0])
        self._merge_sigma_response(filtered_max, first)
        del first
        if cuda_parallel_requested or progress_callback is not None:
            self._sync_progress_device(image)
        if progress_callback is not None:
            progress_callback(1, total)

        streams = 1
        if cuda_parallel_requested:
            measured_peak = max(
                int(torch.cuda.max_memory_allocated(image.device)) - cuda_baseline,
                int(image.numel() * image.element_size()),
            )
            free_bytes, _total_bytes = torch.cuda.mem_get_info(image.device)
            streams = self._memory_limited_sigma_streams(
                self._requested_cuda_sigma_streams(total - 1),
                int(free_bytes),
                measured_peak,
            )

        completed = 1
        remaining = list(self.sigmas[1:])
        if streams <= 1:
            for sigma in remaining:
                vals = self._calc_sigma_response(image, sigma)
                self._merge_sigma_response(filtered_max, vals)
                del vals
                completed += 1
                if progress_callback is not None:
                    self._sync_progress_device(image)
                    progress_callback(completed, total)
            return filtered_max

        for batch_start in range(0, len(remaining), streams):
            batch = remaining[batch_start : batch_start + streams]
            cuda_streams = [torch.cuda.Stream(device=image.device) for _ in batch]
            responses = []
            for sigma, stream in zip(batch, cuda_streams):
                with torch.cuda.stream(stream):
                    responses.append(self._calc_sigma_response(image, sigma))
            current_stream = torch.cuda.current_stream(image.device)
            for stream in cuda_streams:
                current_stream.wait_stream(stream)
            for vals in responses:
                # ``vals`` was allocated on a worker stream but is consumed by
                # the current stream. Keep its storage alive until that merge
                # has actually completed, not merely until it is enqueued.
                vals.record_stream(current_stream)
                self._merge_sigma_response(filtered_max, vals)
            del responses
            completed += len(batch)
            if progress_callback is not None:
                current_stream.synchronize()
                for done in range(completed - len(batch) + 1, completed + 1):
                    progress_callback(done, total)

        return filtered_max


    def forward(self, image, progress_callback=None):
        """
        Apply Frangi filter on a batch of images.
        Arguments:
            img (torch.Tensor, sequence): Tensor of shape (bs, channels, h, w)
        """
        with torch.inference_mode():
            image = torch.as_tensor(image, dtype=torch.float32, device=self.device)
            frangi_resp = self._calc_frangi_response(image, progress_callback=progress_callback)
            return frangi_resp
