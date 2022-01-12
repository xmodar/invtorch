"""Invertible Batch Normalization Modules"""
import functools

import torch
from torch import nn

from ...autograd.grad_mode import in_backward_mode
from ...utils.parametrizations import NonZero
from .module import Module

__all__ = ['BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d']


class _BatchNorm(nn.modules.batchnorm._BatchNorm, Module):
    """Invertible batchnorm layer"""
    # pylint: disable=abstract-method, protected-access
    num_outputs = 1
    forward = Module.forward

    @functools.wraps(nn.modules.batchnorm._BatchNorm.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.non_zero = NonZero(self.eps / 100)

    def function(self, inputs):  # pylint: disable=arguments-differ
        self._check_input_dim(inputs)
        mean, var = self._get_stats(inputs)
        shape = (1, inputs.shape[1]) + (1, ) * (inputs.ndim - 2)
        out = (inputs - mean.view(shape)) / (var.view(shape) + self.eps).sqrt()
        if self.affine:
            self.weight.data.copy_(self.non_zero(self.weight.data))
            out = self.weight.view(shape) * out + self.bias.view(shape)
        if not in_backward_mode():
            self._update_stats(mean, var, inputs.numel())
        # TODO: match PyTorch's BatchNorm forward()
        return out, mean, var

    def inverse(self, outputs, mean, var):  # pylint: disable=arguments-differ
        size = (1, outputs.shape[1]) + (1, ) * (outputs.ndim - 2)
        if self.affine:
            outputs = (outputs - self.bias.view(size)) / self.weight.view(size)
        return outputs * (var.view(size) + self.eps).sqrt() + mean.view(size)

    def _get_stats(self, inputs):
        if self.training or not self.track_running_stats:
            dim = tuple(set(range(inputs.ndim)) - {1})
            var, mean = torch.var_mean(inputs.detach(), dim, unbiased=False)
        else:
            var, mean = self.running_var, self.running_mean
        return mean, var

    def _update_stats(self, mean, var, inputs_numel):
        if self.training and self.track_running_stats:
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:
                factor = 1 / self.num_batches_tracked
            else:
                factor = self.momentum
            unbias = inputs_numel / (inputs_numel - var.numel())
            mean, var = factor * mean, factor * unbias * var
            self.running_mean.mul_(1 - factor).add_(mean)
            self.running_var.mul_(1 - factor).add_(var)

    def extra_repr(self):
        return f'{super().extra_repr()}, {Module.extra_repr(self)}'


class BatchNorm1d(nn.BatchNorm1d, _BatchNorm):
    """Invertible 1D convolution"""


class BatchNorm2d(nn.BatchNorm2d, _BatchNorm):
    """Invertible 2D convolution"""


class BatchNorm3d(nn.BatchNorm3d, _BatchNorm):
    """Invertible 3D convolution"""
