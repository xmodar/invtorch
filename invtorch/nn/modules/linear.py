"""Linear Invertible Modules"""
import functools

from torch import nn
from torch.nn import functional as F

from ...utils.tools import requires_grad
from .module import WrapperModule

__all__ = ['Linear']


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

    def function(self, inputs, *, strict_forward=False, saved=()):
        """Compute the outputs of the function"""
        # pylint: disable=arguments-differ
        if 0 in saved:
            return None
        outputs = F.linear(inputs, self.weight, self.bias)
        if strict_forward:
            requires_grad(outputs, any=(inputs, self.weight, self.bias))
        return outputs

    def inverse(self, outputs, *, strict_forward=False, saved=()):
        """Compute the inputs of the function"""
        # pylint: disable=arguments-differ
        if 0 in saved:
            return None
        if self.bias is not None:
            outputs = outputs - self.bias
        inputs = F.linear(outputs, self.weight.pinverse())
        if strict_forward:
            requires_grad(inputs, any=(outputs, self.weight, self.bias))
        return inputs
