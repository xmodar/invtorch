"""InvTorch: Basic Invertible Modules https://github.com/xmodar/invtorch"""
import functools

import torch
from torch import nn
from torch.nn import functional as F

from .core import InvertibleModule
from .utils import requires_grad

__all__ = ['InvertibleLinear']


class InvertibleLinear(InvertibleModule):
    """Invertible linear module"""
    @functools.wraps(torch.nn.Linear.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = nn.Linear(*args, **kwargs)

    # pylint: disable=arguments-differ, unused-argument
    def function(self, inputs, strict_forward=False):
        """Compute the outputs of the function"""
        outputs = F.linear(inputs, self.weight, self.bias)
        if strict_forward:
            requires_grad(outputs, any=(inputs, self.weight, self.bias))
        return outputs

    def inverse(self, outputs, saved=()):
        """Compute the inputs of the function"""
        if self.bias is not None:
            outputs = outputs - self.bias
        return F.linear(outputs, self.weight.pinverse())

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
