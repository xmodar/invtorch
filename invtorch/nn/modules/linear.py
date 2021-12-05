"""Linear Invertible Modules"""
import functools

from torch import nn
from torch.nn import functional as F

from ...utils.tools import requires_grad
from .module import Module, WrapperModule

__all__ = ['Identity', 'Linear']


class Identity(Module):
    """Identity function"""
    @property
    def reversible(self):
        return True

    def function(self, *inputs, strict=None, saved=()):
        # pylint: disable=unused-argument
        return inputs[0] if len(inputs) == 1 else inputs

    inverse = function

    def forward(self, *inputs, **kwargs):
        kwargs.setdefault('enabled', False)
        return super().forward(*inputs, **kwargs)


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

    def function(self, inputs, *, strict=None, saved=()):
        """Compute the outputs of the function"""
        # pylint: disable=arguments-differ
        if 0 in saved:
            return None
        outputs = F.linear(inputs, self.weight, self.bias)
        if strict:
            requires_grad(outputs, any=(inputs, self.weight, self.bias))
        return outputs

    def inverse(self, outputs, *, strict=None, saved=()):
        """Compute the inputs of the function"""
        # pylint: disable=arguments-differ
        if 0 in saved:
            return None
        if self.bias is not None:
            outputs = outputs - self.bias
        inputs = F.linear(outputs, self.weight.pinverse())
        if strict:
            requires_grad(inputs, any=(outputs, self.weight, self.bias))
        return inputs
