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
        outputs, inputs = self.flat_weight_shape
        assert inputs <= outputs, f'out_channels/groups={outputs} < {inputs}'
        # TODO: assert kernel_size is still invertible given stride & dilation

    def _conv_forward(self, input, weight, bias):
        """perform the forward pass given the inputs and parameters"""
        # pylint: disable=redefined-builtin
        raise NotImplementedError

    @property
    def reversible(self):
        outputs, inputs = self.flat_weight_shape
        return inputs == outputs

    def function(self, inputs):  # pylint: disable=arguments-differ
        # TODO: make input padding an opt-in feature
        input_padding = self.get_input_padding(inputs.shape)
        assert sum(input_padding) == 0, f'inputs need padding: {inputs.shape}'
        return self._conv_forward(inputs, self.weight, self.bias)

    def inverse(self, outputs):  # pylint: disable=arguments-differ
        if self.bias is not None:
            outputs = outputs - self.bias.view(-1, *[1] * self.dim)

        # compute the overlapped inputs
        factor, input_shape = self.flat_weight_shape
        inverse = torch.inverse if self.reversible else torch.pinverse
        weight = self.weight.view(self.groups, -1, input_shape)
        weight = inverse(weight).transpose(-1, -2).reshape(self.weight_shape)
        inputs = self.conv_transpose(outputs, weight)

        # TODO: make kernel overlaps an opt-in feature
        # normalize the overlapping regions
        one = torch.ones((), device=inputs.device, dtype=inputs.dtype)
        outputs = (one / factor).expand(1, *outputs.shape[1:])
        overlaps_weight = one.expand(self.out_channels, 1, *self.kernel_size)
        overlaps = self.conv_transpose(outputs, overlaps_weight)
        return inputs.div_(overlaps)

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
        return len(self.weight_shape) - 2

    @property
    def weight_shape(self):
        """Shape of the weight parameter"""
        return self.parametrizations.weight.original.shape

    @property
    def flat_weight_shape(self):
        """Output and input shapes"""
        shape = self.weight_shape
        return shape[0] // self.groups, shape[1:].numel()

    def extra_repr(self):
        return f'{super().extra_repr()}, {Module.extra_repr(self)}'


class Conv1d(nn.Conv1d, _ConvNd):
    """Invertible 1D convolution layer"""
    forward = Module.forward


class Conv2d(nn.Conv2d, _ConvNd):
    """Invertible 2D convolution layer"""
    forward = Module.forward


class Conv3d(nn.Conv3d, _ConvNd):
    """Invertible 3D convolution layer"""
    forward = Module.forward
