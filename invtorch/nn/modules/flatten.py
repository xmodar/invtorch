"""Invertible Flatten Modules"""
from torch import nn

from ...autograd.grad_mode import in_dry_mode
from .module import Module

__all__ = ['Flatten']


class Flatten(nn.Flatten, Module):
    """Invertible flatten layer"""
    reversible = True
    forward = Module.forward
    num_inputs = num_outputs = 1

    def __init__(self, start_dim=1, end_dim=-1, dims=None):
        super().__init__(start_dim, end_dim)
        self.dims = dims

    def function(self, inputs):  # pylint: disable=arguments-differ
        def check(dim):
            rem, out = divmod(dim, inputs.dim())
            assert abs(rem) < 2, f'{dim} is out of range [{inputs.dim()}]'
            return out

        start_dim, end_dim = check(self.start_dim), check(self.end_dim)
        end = None if end_dim == inputs.dim() else end_dim + 1
        dims = inputs.shape[start_dim:end]
        inputs = inputs.flatten(start_dim, end_dim)
        return inputs.clone() if in_dry_mode() else inputs, dims

    def inverse(self, outputs, dims=None):  # pylint: disable=arguments-differ
        if dims is None:
            dims = self.dims
        assert dims is not None, 'must provide `dims` or set `self.dims`'
        outputs = outputs.unflatten(self.start_dim, dims)
        return outputs.clone() if in_dry_mode() else outputs

    def extra_repr(self):
        return f'{super().extra_repr()}, {Module.extra_repr(self)}'
