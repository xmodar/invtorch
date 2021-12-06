"""Invertible Flatten Modules"""
from .module import Module

__all__ = ['Flatten']


class Flatten(Module):
    """Invertible flatten operation"""
    def __init__(self, start_dim=1, end_dim=-1, unflattened_size=None):
        super().__init__()
        self.start_dim, self.end_dim = start_dim, end_dim
        self.unflattened_size = unflattened_size

    def function(self, inputs, *, strict=None):
        # pylint: disable=arguments-differ, unused-argument
        end_dim = self.end_dim
        if end_dim < 0:
            end_dim += inputs.dim()
        if self.end_dim == inputs.dim():
            end_dim = None
        else:
            end_dim += 1
        unflattened_size = inputs.shape[self.start_dim:end_dim]
        if self.unflattened_size is not None:
            assert tuple(self.unflattened_size) == unflattened_size, 'mismatch'
        outputs = inputs.flatten(self.start_dim, self.end_dim)
        if strict:
            outputs.requires_grad_(inputs.requires_grad)
        return outputs, unflattened_size

    def inverse(self, outputs, unflattened_size=None, *, strict=None):
        # pylint: disable=arguments-differ, unused-argument
        if unflattened_size is None:
            unflattened_size = self.unflattened_size
        inputs = outputs.unflatten(self.start_dim, unflattened_size)
        if strict:
            inputs.requires_grad_(outputs.requires_grad)
        return inputs

    num_function_outputs = 1

    @property
    def reversible(self):
        return self.unflattened_size is not None
