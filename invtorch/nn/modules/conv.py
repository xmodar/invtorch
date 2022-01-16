"""Invertible Convolutions"""
import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize as P

from ...utils.parametrizations import ScaledOrthogonal
from .module import Module

__all__ = ['Conv1d', 'Conv2d', 'Conv3d']


class _ConvNd(nn.modules.conv._ConvNd, Module):
    """Invertible convolution layer"""
    # pylint: disable=protected-access

    @functools.wraps(nn.modules.conv._ConvNd.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        orthogonal = ScaledOrthogonal(self.weight, flatten=1)
        P.register_parametrization(self, 'weight', orthogonal, unsafe=True)
        self.ceil_mode = False
        outputs, inputs = self.flat_weight_shape
        assert inputs <= outputs, f'out_channels/groups={outputs} < {inputs}'

    @property
    def reversible(self):
        outputs, inputs = self.flat_weight_shape
        return inputs == outputs

    def _conv_forward(self, input, weight, bias):
        """perform the forward pass given the inputs and parameters"""
        # pylint: disable=redefined-builtin
        raise NotImplementedError

    def function(self, inputs):  # pylint: disable=arguments-differ
        # TODO: make input padding an opt-in feature
        input_padding = self.get_input_padding(inputs.shape)
        assert sum(input_padding) == 0, f'inputs need padding: {inputs.shape}'
        return self._conv_forward(inputs, self.weight, self.bias)

    def inverse(self, outputs):  # pylint: disable=arguments-differ
        if self.bias is not None:
            outputs = outputs - self.bias.view(-1, *[1] * self.dim)
        factor, input_numel = self.flat_weight_shape
        weight = self.weight.view(self.groups, -1, input_numel)
        inverse = torch.inverse if self.reversible else torch.pinverse
        weight = inverse(weight).transpose(-1, -2).reshape(self.weight_shape)
        inputs = self.conv_transpose(outputs, weight)
        if self.kernel_size != inputs.shape[2:] and self.overlaps:
            one = torch.ones((), device=inputs.device, dtype=inputs.dtype)
            weight = one.expand(self.out_channels, 1, *self.kernel_size)
            outputs = (one / factor).expand(1, *outputs.shape[1:])
            inputs /= self.conv_transpose(outputs, weight)
        return inputs

    def get_input_padding(self, input_size):
        """Get the input padding given the input size"""
        def rule(size, kernel_size, stride, padding, dilation):
            margin = 2 * padding - dilation * (kernel_size - 1) - 1
            return size - ((size + margin) // stride) * stride + margin

        input_size = tuple(input_size)[-self.dim:]
        args = self.kernel_size, self.stride, self.padding, self.dilation
        return tuple(map(lambda x: rule(*x), zip(input_size, *args)))

    def conv_transpose(self, inputs, weight):
        """Compute conv_transpose"""
        args = None, self.stride, self.padding, 0, self.groups, self.dilation
        return getattr(F, f'conv_transpose{self.dim}d')(inputs, weight, *args)

    @property
    def dim(self):
        """Number of dimensions"""
        return len(self.kernel_size)

    @property
    def weight_shape(self):
        """Shape of the weight parameter"""
        return torch.Size([
            self.out_channels,
            self.in_channels // self.groups,
            *self.kernel_size,
        ])

    @property
    def flat_weight_shape(self):
        """Shape of the flat weight parameter per group"""
        shape = self.weight_shape
        return torch.Size([shape[0] // self.groups, shape[1:].numel()])

    @property
    def overlaps(self):
        """Whether kernel positions overlap during the convolution"""
        def rule(kernel_size, stride, dilation):
            return kernel_size == stride and 1 in (dilation, kernel_size)

        args = zip(self.kernel_size, self.stride, self.dilation)
        return not all(map(lambda x: rule(*x), args))

    def extra_repr(self):
        return f'{super().extra_repr()}, {Module.extra_repr(self)}'


class Conv1d(nn.Conv1d, _ConvNd):
    """Invertible 1D convolution layer"""
    forward = Module.forward

    @functools.wraps(nn.Conv1d.__init__)
    def __init__(self, *args, ceil_mode=False, overlaps=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ceil_mode = bool(ceil_mode)
        if self.ceil_mode:
            raise NotImplementedError('ceil_mode')
        assert overlaps or not self.overlaps, (
            'This convolution can overlap (slow and requires more memory). '
            'Either set `stride` equal to `kernel_size` and `dilation=1` '
            'or set `overlaps=True` to force this manually.')


class Conv2d(nn.Conv2d, _ConvNd):
    """Invertible 2D convolution layer"""
    forward = Module.forward
    __init__ = functools.wraps(nn.Conv2d.__init__)(Conv1d.__init__)


class Conv3d(nn.Conv3d, _ConvNd):
    """Invertible 3D convolution layer"""
    forward = Module.forward
    __init__ = functools.wraps(nn.Conv3d.__init__)(Conv1d.__init__)
