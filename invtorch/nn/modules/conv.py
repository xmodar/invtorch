"""Invertible Convolutions"""
import functools

import torch
import torch.nn.functional as F
from torch import nn

from ...utils.tools import requires_grad
from .module import WrapperModule

__all__ = ['Conv1d', 'Conv2d', 'Conv3d']


class _ConvNd(WrapperModule):
    """Invertible convolution"""
    wrapped_type = nn.modules.conv._ConvNd  # pylint: disable=protected-access

    def __init__(self, module):
        super().__init__(module)
        outputs, inputs = self.flat_weight_shape
        assert inputs <= outputs, f'out_channels/groups={outputs} < {inputs}'
        # TODO: assert kernel_size is still invertible given stride & dilation

    @property
    def reversible(self):
        outputs, inputs = self.flat_weight_shape
        return inputs == outputs, f'out_channels/groups={outputs} != {inputs}'

    def function(self, inputs, *, strict=None):
        # pylint: disable=arguments-differ
        # TODO: make input padding an opt-in feature
        input_padding = self.get_input_padding(inputs.shape)
        assert sum(input_padding) == 0, f'inputs need padding: {inputs.shape}'
        outputs = self.module.forward(inputs)
        if strict:
            requires_grad(outputs, any=(inputs, self.weight, self.bias))
        return outputs

    def inverse(self, outputs, *, strict=None):
        # pylint: disable=arguments-differ
        old_outputs = outputs
        if self.bias is not None:
            outputs = outputs - self.bias.view(-1, *[1] * self.dim)

        # compute the overlapped inputs
        factor, input_shape = self.flat_weight_shape
        weight = self.weight.view(self.groups, -1, input_shape)
        weight = weight.pinverse().transpose(-1, -2).reshape_as(self.weight)
        inputs = self.conv_transpose(outputs, weight)

        # TODO: make kernel overlaps an opt-in feature
        # normalize the overlapping regions
        one = torch.ones((), device=inputs.device, dtype=inputs.dtype)
        outputs = (one / factor).expand(1, *outputs.shape[1:])
        overlaps_weight = one.expand(self.out_channels, 1, *self.kernel_size)
        overlaps = self.conv_transpose(outputs, overlaps_weight)
        inputs = inputs.div_(overlaps)

        if strict:
            requires_grad(inputs, any=(old_outputs, self.weight, self.bias))
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
        return self.weight.dim() - 2

    @property
    def flat_weight_shape(self):
        """Output and input shapes"""
        shape = self.weight.shape
        return shape[0] // self.groups, shape[1:].numel()


class Conv1d(_ConvNd):
    """Invertible 1D convolution"""
    wrapped_type = nn.Conv1d

    @functools.wraps(nn.Conv1d.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(nn.Conv1d(*args, **kwargs))


class Conv2d(_ConvNd):
    """Invertible 2D convolution"""
    wrapped_type = nn.Conv2d

    @functools.wraps(nn.Conv2d.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(nn.Conv2d(*args, **kwargs))


class Conv3d(_ConvNd):
    """Invertible 3D convolution"""
    wrapped_type = nn.Conv3d

    @functools.wraps(nn.Conv3d.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(nn.Conv3d(*args, **kwargs))
