"""Invertible Batch Normalization Modules"""
import functools

import torch
from torch import nn

from ...autograd.grad_mode import in_backward_mode
from ...utils.parametrizations import NonZero
from .module import WrapperModule

__all__ = ['BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d']


class _BatchNorm(WrapperModule):
    """Invertible batchnorm layer (inverse doesn't update running stats)"""
    wrapped_type = nn.modules.batchnorm._BatchNorm  # pylint: disable=protected-access
    num_outputs = 1

    def __init__(self, module):
        super().__init__(module)
        self.non_zero = NonZero(self.eps / 100)

    def function(self, inputs):  # pylint: disable=arguments-differ
        """Perform the forward pass"""
        self._check_input_dim(inputs)
        mean, var = self._get_stats(inputs)
        shape = (1, inputs.shape[1]) + (1, ) * (inputs.ndim - 2)
        out = (inputs - mean.view(shape)) / (var.view(shape) + self.eps).sqrt()
        if self.affine:
            self.weight.data.copy_(self.non_zero(self.weight.data))
            out = self.weight.view(shape) * out + self.bias.view(shape)
        if not in_backward_mode():
            self._update_stats(mean, var, inputs.numel())
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
            self.module.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:
                factor = 1 / self.num_batches_tracked
            else:
                factor = self.momentum
            unbias = inputs_numel / (inputs_numel - var.numel())
            mean, var = factor * mean, factor * unbias * var
            self.running_mean.mul_(1 - factor).add_(mean)
            self.running_var.mul_(1 - factor).add_(var)


class BatchNorm1d(_BatchNorm):
    """Invertible 1D convolution"""
    wrapped_type = nn.BatchNorm1d

    @functools.wraps(nn.BatchNorm1d.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(nn.BatchNorm1d(*args, **kwargs))


class BatchNorm2d(_BatchNorm):
    """Invertible 2D convolution"""
    wrapped_type = nn.BatchNorm2d

    @functools.wraps(nn.BatchNorm2d.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(nn.BatchNorm2d(*args, **kwargs))


class BatchNorm3d(_BatchNorm):
    """Invertible 3D convolution"""
    wrapped_type = nn.BatchNorm3d

    @functools.wraps(nn.BatchNorm3d.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(nn.BatchNorm3d(*args, **kwargs))
