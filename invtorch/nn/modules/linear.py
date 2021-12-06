"""Linear Invertible Modules"""
import functools

from torch import nn
from torch.nn import functional as F

from ...utils.tools import requires_grad
from .module import Module, WrapperModule

__all__ = ['Identity', 'Linear']


class Identity(Module):
    """Identity function"""
    reversible = True

    def forward(self, *inputs, **kwargs):
        return inputs[0] if len(inputs) == 1 else inputs

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

    def function(self, inputs, *, strict=None):
        """Compute the outputs of the function"""
        # pylint: disable=arguments-differ
        outputs = F.linear(inputs, self.weight, self.bias)
        if strict:
            requires_grad(outputs, any=(inputs, self.weight, self.bias))
        return outputs

    def inverse(self, outputs, *, strict=None):
        """Compute the inputs of the function"""
        # pylint: disable=arguments-differ
        if self.bias is not None:
            outputs = outputs - self.bias
        inputs = F.linear(outputs, self.weight.pinverse())
        if strict:
            requires_grad(inputs, any=(outputs, self.weight, self.bias))
        return inputs
