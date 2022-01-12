"""Linear Invertible Modules"""
import functools

from torch import nn
from torch.nn import functional as F

from .module import Module

__all__ = ['Identity', 'Linear']


class Identity(nn.Identity, Module):
    """Identity module"""
    reversible = True

    def forward(self, *args, **kwargs):  # pylint: disable=unused-argument
        return args[0] if len(args) == 1 else args

    function = inverse = forward


class Linear(nn.Linear, Module):
    """Invertible linear module"""
    forward = Module.forward

    @functools.wraps(nn.Linear.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.in_features <= self.out_features, 'few out_features'

    @property
    def reversible(self):
        return self.in_features == self.out_features

    def function(self, inputs):  # pylint: disable=arguments-differ
        """Compute the outputs of the function"""
        return super().forward(inputs)

    def inverse(self, outputs):  # pylint: disable=arguments-differ
        """Compute the inputs of the function"""
        if self.bias is not None:
            outputs = outputs - self.bias
        return F.linear(outputs, self.weight.pinverse())

    def extra_repr(self):
        return f'{super().extra_repr()}, {Module.extra_repr(self)}'
