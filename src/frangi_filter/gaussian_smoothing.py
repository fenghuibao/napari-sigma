import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F

"""
Gaussian smoothing with Pytorch.
Source:
https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
https://dsp.stackexchange.com/questions/78280/are-scipy-second-order-gaussian-derivatives-correct
https://github.com/ilyas-sid/SoftFrangiFilter2D
"""


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2, order=0, device=None):
        super().__init__()
        self.dim = dim
        self.order = order
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        self.kernel_size = list(kernel_size)
        self.sigma = [float(v) for v in sigma]
        self.padding = [int(size // 2) for size in self.kernel_size]
        self.groups = channels

        if dim not in (1, 2, 3):
            raise RuntimeError(
                f'Only 1, 2 and 3 dimensions are supported. Received {dim}.'
            )

        derivative_axis = None
        if order == 'x':
            derivative_axis = dim - 1
        elif order == 'y':
            derivative_axis = dim - 2
        elif order == 'z':
            derivative_axis = dim - 3
        self.derivative_axis = derivative_axis

        for axis in range(dim):
            kernel_1d = self._make_axis_kernel(
                size=self.kernel_size[axis],
                std=self.sigma[axis],
                derivative=(axis == derivative_axis),
            )
            weight = self._reshape_axis_kernel(kernel_1d, dim, axis, channels)
            if device is not None:
                weight = weight.to(device)
            self.register_buffer(f'weight_{axis}', weight)

    @staticmethod
    def _make_axis_kernel(size, std, derivative):
        pad = size // 2
        coords = torch.arange(-pad, pad + 1, dtype=torch.float32)
        kernel = 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(coords / std) ** 2 / 2)
        kernel = kernel / torch.sum(kernel)
        if derivative:
            kernel = -coords / (std**2) * kernel
        return kernel

    @staticmethod
    def _reshape_axis_kernel(kernel_1d, dim, axis, channels):
        shape = [1] * dim
        shape[axis] = kernel_1d.numel()
        kernel = kernel_1d.view(1, 1, *shape)
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        kernel.requires_grad = False
        return kernel

    def _pad_for_axis(self, x, axis):
        pad = self.padding[axis]
        if pad == 0:
            return x
        if self.dim == 1:
            return F.pad(x, (pad, pad), mode='reflect')
        if self.dim == 2:
            if axis == 0:
                return F.pad(x, (0, 0, pad, pad), mode='reflect')
            return F.pad(x, (pad, pad, 0, 0), mode='reflect')
        if axis == 0:
            return F.pad(x, (0, 0, 0, 0, pad, pad), mode='reflect')
        if axis == 1:
            return F.pad(x, (0, 0, pad, pad, 0, 0), mode='reflect')
        return F.pad(x, (pad, pad, 0, 0, 0, 0), mode='reflect')

    def _convolve_axis(self, x, axis):
        if axis == self.derivative_axis:
            return self._convolve_derivative_axis(x, axis)

        weight = getattr(self, f'weight_{axis}')
        x = self._pad_for_axis(x, axis)
        if self.dim == 1:
            return F.conv1d(x, weight=weight, groups=self.groups)
        if self.dim == 2:
            return F.conv2d(x, weight=weight, groups=self.groups)
        return F.conv3d(x, weight=weight, groups=self.groups)

    def _convolve_derivative_axis(self, x, axis):
        """Apply an antisymmetric derivative kernel with exact pair cancellation."""
        pad = self.padding[axis]
        if pad == 0:
            return torch.zeros_like(x)

        padded = self._pad_for_axis(x, axis)
        spatial_axis = x.ndim - self.dim + axis
        axis_size = x.shape[spatial_axis]
        weight = getattr(self, f'weight_{axis}')
        channel_axis = x.ndim - self.dim - 1
        coefficient_shape = [1] * x.ndim
        coefficient_shape[channel_axis] = weight.shape[0]
        output = torch.zeros_like(x)

        weight_index = [slice(None), 0, *([0] * self.dim)]
        left_index = [slice(None)] * padded.ndim
        right_index = [slice(None)] * padded.ndim
        for offset in range(1, pad + 1):
            weight_index[axis + 2] = pad - offset
            coefficient = weight[tuple(weight_index)].reshape(
                tuple(coefficient_shape)
            )
            left_index[spatial_axis] = slice(
                pad - offset,
                pad - offset + axis_size,
            )
            right_index[spatial_axis] = slice(
                pad + offset,
                pad + offset + axis_size,
            )
            output = output + coefficient * (
                padded[tuple(left_index)] - padded[tuple(right_index)]
            )
        return output

    def forward(self, x):
        """
        Apply gaussian filter to input.
        Arguments:
            x (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        out = x
        for axis in range(self.dim):
            out = self._convolve_axis(out, axis)
        return -out
