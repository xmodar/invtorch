"""Linear Invertible Modules"""
import functools

from torch import nn
from torch.nn import functional as F

from .module import Module, WrapperModule

__all__ = ['Identity', 'Linear']


class Identity(Module):
    """Identity function"""
    reversible = True

    def forward(self, *args, **kwargs):
        return args[0] if len(args) == 1 else args

    function = inverse = call_function = call_inverse = forward


class Linear(WrapperModule):
    """Invertible linear module"""
    wrapped_type = nn.Linear

    @functools.wraps(nn.Linear.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(nn.Linear(*args, **kwargs))
        assert self.in_features <= self.out_features, 'few out_features'

    @property
    def reversible(self):
        return self.in_features == self.out_features

    def function(self, inputs, cache=None):
        """Compute the outputs of the function"""
        # pylint: disable=arguments-differ, unused-argument
        return self.module.forward(inputs)

    def inverse(self, outputs, cache=None):
        """Compute the inputs of the function"""
        # pylint: disable=arguments-differ, unused-argument
        if self.bias is not None:
            outputs = outputs - self.bias
        return F.linear(outputs, self.weight.pinverse())
