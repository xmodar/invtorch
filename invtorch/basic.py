"""InvTorch: Basic Invertible Modules https://github.com/xmodar/invtorch"""
import functools

import torch
from torch import nn
import torch.nn.functional as F

from .core import InvertibleModule

__all__ = ['InvertibleLinear']


class InvertibleLinear(InvertibleModule):
    """Invertible linear module"""
    @functools.wraps(torch.nn.Linear.__init__)
    def __init__(self, *args, invertible=True, checkpoint=True, **kwargs):
        super().__init__(invertible=invertible, checkpoint=checkpoint)
        self.model = nn.Linear(*args, **kwargs)

    def function(self, inputs):  # pylint: disable=arguments-differ
        """Compute the outputs of the function"""
        outputs = F.linear(inputs, self.weight, self.bias)
        requires_grad = self.do_require_grad(inputs, self.weight, self.bias)
        return outputs.requires_grad_(requires_grad)

    def inverse(self, outputs):  # pylint: disable=arguments-differ
        """Compute the inputs of the function"""
        if self.bias is not None:
            outputs = outputs - self.bias
        return F.linear(outputs, self.weight.pinverse())

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
