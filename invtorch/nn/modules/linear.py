"""Linear Invertible Modules"""
import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize as P

from ...utils.parametrizations import ScaledOrthogonal
from .module import Module

__all__ = ['Identity', 'Linear']


class Identity(nn.Identity, Module):
    """Identity module"""
    reversible = True

    def forward(self, *args, **kwargs):  # pylint: disable=unused-argument
        return args[0] if len(args) == 1 else args

    function = inverse = forward


class Linear(nn.Linear, Module):
    """Invertible linear layer"""
    forward = Module.forward

    @functools.wraps(nn.Linear.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.in_features <= self.out_features, 'few out_features'
        orthogonal = ScaledOrthogonal(self.weight)
        P.register_parametrization(self, 'weight', orthogonal, unsafe=True)

    @property
    def reversible(self):
        return self.in_features == self.out_features

    def function(self, inputs):  # pylint: disable=arguments-differ
        """Compute the outputs of the function"""
        return F.linear(inputs, self.weight, self.bias)

    def inverse(self, outputs):  # pylint: disable=arguments-differ
        """Compute the inputs of the function"""
        if self.bias is not None:
            outputs = outputs - self.bias
        inverse = torch.inverse if self.reversible else torch.pinverse
        return F.linear(outputs, inverse(self.weight))

    def extra_repr(self):
        return f'{super().extra_repr()}, {Module.extra_repr(self)}'
