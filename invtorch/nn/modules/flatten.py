"""Invertible Flatten Modules"""
from .module import Module

__all__ = ['Flatten']


class Flatten(Module):
    """Invertible flatten operation"""
    def __init__(self, start_dim=1, end_dim=-1, dims=None):
        super().__init__()
        self.start_dim, self.end_dim, self.dims = start_dim, end_dim, dims

    def function(self, inputs, cache=None):
        # pylint: disable=arguments-differ
        start_rem, start_dim = divmod(self.start_dim, inputs.dim())
        assert abs(start_rem) < 2, '`self.start_dim` is out of range'
        end_rem, end_dim = divmod(self.end_dim, inputs.dim())
        assert abs(end_rem) < 2, '`self.end_dim` is out of range'
        if cache is not None:
            last = None if end_dim == inputs.dim() else end_dim + 1
            cache['dims'] = inputs.shape[start_dim:last]
        return inputs.flatten(start_dim, end_dim)

    def inverse(self, outputs, dims=None, cache=None):
        # pylint: disable=arguments-differ, unused-argument
        if dims is None:
            dims = self.dims
        assert dims is not None, 'must provide `dims` or set `self.dims`'
        return outputs.unflatten(self.start_dim, dims)
