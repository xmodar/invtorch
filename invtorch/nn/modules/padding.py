"""Invertible Padding Modules"""
from torch.nn import functional as F

from .module import Module

__all__ = ['Pad']


class Pad(Module):
    """Invertible padding layer"""
    num_outputs = 1

    def __init__(self, mode='constant', value=0):
        super().__init__()
        self.mode, self.value = mode, value

    def function(self, inputs, pad=None):  # pylint: disable=arguments-differ
        if not self.passthrough(pad):
            inputs = F.pad(inputs, pad, self.mode, self.value)
        return inputs, pad

    def inverse(self, outputs, pad):  # pylint: disable=arguments-differ
        if self.passthrough(pad):
            return outputs
        to_slice = lambda *args: slice(*(None if x == 0 else x for x in args))
        index = tuple(to_slice(s, -e) for s, e in zip(pad[::2], pad[1::2]))
        return outputs[(..., *reversed(index))], pad

    @staticmethod
    def multiple(size, stride):
        """Get the required padding to make `size` a multiple of `stride`"""
        assert len(size) == len(stride)
        remainders = reversed([-d % s for d, s in zip(size, stride)])
        return tuple(x for r in remainders for x in (r // 2, -(-r // 2)))

    @staticmethod
    def passthrough(pad=None):
        """Whether padding is not necessary"""
        return pad is None or all(x == 0 for x in pad)
