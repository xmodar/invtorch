"""Invertible Flatten Modules"""
from torch import nn

from .module import Module

__all__ = ['Flatten']


class Flatten(nn.Flatten, Module):
    """Invertible flatten operation"""
    num_outputs = 1
    forward = Module.forward

    def __init__(self, start_dim=1, end_dim=-1, dims=None):
        super().__init__(start_dim, end_dim)
        self.dims = dims
        self.checkpoint = False

    def function(self, inputs):  # pylint: disable=arguments-differ
        def check(dim):
            rem, out = divmod(dim, inputs.dim())
            assert abs(rem) < 2, f'{dim} is out of range [{inputs.dim()}]'
            return out

        start_dim, end_dim = check(self.start_dim), check(self.end_dim)
        end = None if end_dim == inputs.dim() else end_dim + 1
        dims = inputs.shape[start_dim:end]
        return inputs.flatten(start_dim, end_dim), dims

    def inverse(self, outputs, dims=None):  # pylint: disable=arguments-differ
        if dims is None:
            dims = self.dims
        assert dims is not None, 'must provide `dims` or set `self.dims`'
        return outputs.unflatten(self.start_dim, dims)

    @property
    def reversible(self):
        return self.dims is not None

    def extra_repr(self):
        return f'{super().extra_repr()}, {Module.extra_repr(self)}'
