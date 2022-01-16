"""Parametrization Modules"""
import torch
from torch import nn
from torch.nn.utils.parametrizations import _OrthMaps, _Orthogonal

__all__ = ['NonZero', 'Orthogonal', 'ScaledOrthogonal']


class NonZero(nn.Module):
    """Parameterization to force the values to be nonzero"""
    def __init__(self, preserve_sign=True):
        super().__init__()
        self.preserve_sign = preserve_sign

    def forward(self, inputs):
        """Perform the forward pass"""
        return self.call(inputs, self.preserve_sign)

    right_inverse = forward

    @staticmethod
    def call(inputs, preserve_sign=True):
        """Force values to be nonzero"""
        eps = torch.finfo(inputs.dtype).eps * 1e2
        eps_t = torch.tensor(eps, dtype=inputs.dtype, device=inputs.device)
        if preserve_sign:
            eps_t = torch.where(inputs < 0, -eps_t, eps_t)
        return inputs.where(inputs.detach().abs() > eps, eps_t)


class Orthogonal(_Orthogonal):
    """Orthogonal or unitary parametrization for matrices"""
    def __init__(self, weight, view=None, strategy=None, fast=True):
        if view is not None:
            weight = weight.data.view(view)
        assert weight.dim() > 1, f'viewed tensor is {weight.dim()}D (< 2D)'
        if strategy is None:
            if weight.shape[-2] == weight.shape[-1] or weight.is_complex():
                strategy = 'matrix_exp'
            else:
                strategy = 'householder'
        orth_enum = getattr(_OrthMaps, strategy, None)
        if orth_enum is None:
            maps = {x.name for x in _OrthMaps}
            raise ValueError(f'strategy={strategy} not in {maps}')
        super().__init__(weight, orth_enum, use_trivialization=fast)
        self.view = view

    def call(self, function, weight):
        """calls a function on a tensor and views it if necessary"""
        if self.view is not None:
            return function(weight.view(self.view)).view_as(weight)
        return function(weight)

    def forward(self, weight):  # pylint: disable=arguments-renamed
        return self.call(super().forward, weight)

    def right_inverse(self, weight):  # pylint: disable=arguments-renamed
        return self.call(super().right_inverse, weight)


class ScaledOrthogonal(Orthogonal):
    """Scaled orthogonal parametrization for matrices"""
    def call(self, function, weight):
        def function_(matrix):
            eps = torch.finfo(matrix.dtype).eps * 1e2
            dim = -1 if weight.shape[-1] > weight.shape[-2] else -2
            norm = matrix.norm(2, dim, keepdim=True).clamp_min(eps)
            return function(matrix / norm) * norm

        return super().call(function_, weight)
