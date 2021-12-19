"""Linear Invertible Modules"""
import functools

from torch import nn
from torch.nn import functional as F

from .module import Module, WrapperModule

__all__ = ['Identity', 'Linear']


class Identity(Module):
    """Identity function"""
    reversible = True

    def __init__(self):
        super().__init__()
        self.checkpoint = False

    def function(self, *args):
        return args[0] if len(args) == 1 else args

    def inverse(self, *args):
        return self.function(*args)


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

    def function(self, inputs):
        """Compute the outputs of the function"""
        # pylint: disable=arguments-differ
        return self.module.forward(inputs)

    def inverse(self, outputs):
        """Compute the inputs of the function"""
        # pylint: disable=arguments-differ
        if self.bias is not None:
            outputs = outputs - self.bias
        return F.linear(outputs, self.weight.pinverse())
