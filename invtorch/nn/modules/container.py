"""Invertible Container Modules"""
import functools

from torch import nn

from ...utils.tools import pack
from .module import Module

__all__ = ['Sequential']


class Sequential(nn.Sequential, Module):
    """Sequential Invertible module"""
    @functools.wraps(nn.Sequential.__init__)
    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
        assert all(not x.seed for x in self), 'Sequential cannot maintain rng'

    def function(self, *args):
        extras, counts = [], []
        for layer in self:
            outputs = pack(layer.call_function(*args))
            args = pack(layer.process(outputs))
            counts.append(len(outputs) - len(args))
            extras.extend(outputs[len(args):])
        return (*args, *extras, counts)

    def inverse(self, *args):
        extras, end = [()] * len(args[-1]), -1
        for i, count in enumerate(reversed(args[end]), 1):
            extras[-i] = args[end - count:end]
            end -= count
        args = args[:end]
        for layer in reversed(self):
            args = pack(layer.call_inverse(*args, *extras.pop()))
        return args[0] if len(args) == 1 else args

    def process(self, outputs):
        process = self[-1].process if self else super().process
        return process(outputs[:-sum(outputs[-1]) - 1])

    call_function, call_inverse = function, inverse

    @property
    def num_outputs(self):
        """end index to slice the outputs of `call_function()`"""
        return self[-1].num_outputs if self else None

    @property
    def reversible(self):
        return all(layer.reversible for layer in self)

    def reverse(self, mode=None):
        if self.reversed != super().reverse(mode).reversed:
            layers = [layer.reverse() for layer in self]
            for i, layer in enumerate(reversed(layers)):
                self[i] = layer
        return self
