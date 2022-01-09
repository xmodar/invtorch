"""Parametrization Modules"""
import torch
from torch import nn

__all__ = ['NonZero']


class NonZero(nn.Module):
    """Parameterization to force the values to be nonzero"""
    def __init__(self, eps=1e-5, preserve_sign=True):
        super().__init__()
        self.eps, self.preserve_sign = eps, preserve_sign

    def forward(self, inputs):
        """Perform the forward pass"""
        eps = torch.tensor(self.eps, dtype=inputs.dtype, device=inputs.device)
        if self.preserve_sign:
            eps = torch.where(inputs < 0, -eps, eps)
        return inputs.where(inputs.detach().abs() > self.eps, eps)
