"""Invertible Convolution Modules"""
import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize as P

from ...utils.parametrizations import ScaledOrthogonal
from .module import Module
from .padding import Pad

__all__ = ['Conv1d', 'Conv2d', 'Conv3d']


class _ConvNd(nn.modules.conv._ConvNd, Module):
    """Invertible convolution layer"""
    # pylint: disable=protected-access
    num_inputs = num_outputs = 1

    @functools.wraps(nn.modules.conv._ConvNd.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inputs = self.weight.shape[1:].numel()
        outputs = self.out_channels // self.groups
        assert inputs <= outputs, f'out_channels/groups={outputs} < {inputs}'
        self.conv_transpose = functools.partial(
            getattr(F, f'conv_transpose{len(self.kernel_size)}d'),
            bias=None,
            stride=self.stride,
            padding=self.padding,
            output_padding=0,
            groups=self.groups,
            dilation=self.dilation,
        )
        param = ScaledOrthogonal(self.weight, (self.groups, outputs, inputs))
        P.register_parametrization(self, 'weight', param, unsafe=True)
        self._reversible = inputs == outputs
        self.pad = self.patchwise = None

    def _init_mode(self, ceil_mode, patchwise):
        self.pad = Pad() if ceil_mode else None
        args = self.kernel_size, self.stride, self.dilation

        def fits(kernel_size, stride, dilation):
            return stride <= kernel_size and (1 in (dilation, kernel_size))

        assert all(map(lambda x: fits(*x), zip(*args))), (
            'Bad arguments! Set `stride` <= `kernel_size` and `dilation=1')

        def tiles(kernel_size, stride, dilation):
            return stride == kernel_size and (1 in (dilation, kernel_size))

        self.patchwise = all(map(lambda x: tiles(*x), zip(*args)))
        assert not patchwise or self.patchwise, (
            'The `patchwise` mode is enabled by default for efficiency. '
            'Either set `stride` == `kernel_size` and `dilation=1` '
            'or set `patchwise=False` to force using the slow mode.')

    @property
    def reversible(self):
        return not self.ceil_mode and self._reversible

    def _conv_forward(self, input, weight, bias):
        """perform the forward pass given the inputs and parameters"""
        # pylint: disable=redefined-builtin
        raise NotImplementedError

    def function(self, inputs):  # pylint: disable=arguments-differ
        def rule(size, kernel_size, padding, dilation):
            effective_kernel_size = dilation * (kernel_size - 1) + 1
            return size + 2 * padding - effective_kernel_size

        pad = self.pad if self.ceil_mode else Pad
        args = self.kernel_size, self.padding, self.dilation
        size = map(lambda x: rule(*x), zip(inputs.shape[2:], *args))
        padding = pad.multiple(tuple(size), self.stride)
        if self.ceil_mode:
            inputs, padding = self.pad.call_function(inputs, padding)
        else:
            assert pad.passthrough(padding), (
                f'`inputs` of shape {inputs.shape} needs padding {padding}. '
                'Either do it manually or use `ceil_mode=True`.')
            padding = None
        return self._conv_forward(inputs, self.weight, self.bias), padding

    def inverse(self, outputs, padding=None):
        # pylint: disable=arguments-differ
        shape = (1, *outputs.shape[1:])
        if self.bias is not None:
            outputs = outputs - self.bias.view(-1, *[1] * (len(shape) - 2))
        outputs = self.conv_transpose(outputs, self.inverse_weight)
        if not self.patchwise and self.kernel_size != outputs.shape[2:]:
            one = torch.ones((), device=outputs.device, dtype=outputs.dtype)
            weight = one.expand(self.out_channels, 1, *self.kernel_size)
            inputs = (one / (self.out_channels // self.groups)).expand(shape)
            outputs /= self.conv_transpose(inputs, weight)
        if self.ceil_mode:
            outputs = self.pad.call_inverse(outputs, padding)
        else:
            assert padding is None, 'cannot use `pad` when not in `ceil_mode`'
        return outputs

    @property
    def inverse_weight(self):
        """The inverse of the weight parameter"""
        weight = self.weight
        inputs = weight.shape[1:].numel()
        outputs = self.out_channels // self.groups
        flat_weight = self.weight.view(self.groups, outputs, inputs)
        inverse = torch.inverse if inputs == outputs else torch.pinverse
        return inverse(flat_weight).transpose(-1, -2).reshape(weight.shape)

    @property
    def ceil_mode(self):
        """Whether to pad inputs (same as `ceil_mode` in pooling layers)"""
        return self.pad is not None

    def extra_repr(self):
        return f'{super().extra_repr()}, {Module.extra_repr(self)}'


class Conv1d(nn.Conv1d, _ConvNd):
    """Invertible 1D convolution layer"""
    forward = Module.forward

    @functools.wraps(nn.Conv1d.__init__)
    def __init__(self, *args, ceil_mode=False, patchwise=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_mode(ceil_mode, patchwise)


class Conv2d(nn.Conv2d, _ConvNd):
    """Invertible 2D convolution layer"""
    forward = Module.forward

    @functools.wraps(nn.Conv2d.__init__)
    def __init__(self, *args, ceil_mode=False, patchwise=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_mode(ceil_mode, patchwise)


class Conv3d(nn.Conv3d, _ConvNd):
    """Invertible 3D convolution layer"""
    forward = Module.forward

    @functools.wraps(nn.Conv3d.__init__)
    def __init__(self, *args, ceil_mode=False, patchwise=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_mode(ceil_mode, patchwise)
