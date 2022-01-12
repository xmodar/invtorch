"""Invertible Container Modules"""
from torch import nn

from ...utils.tools import pack
from .module import Module

__all__ = ['Sequential']


class Sequential(nn.Sequential, Module):
    """Sequential Invertible module"""
    forward = Module.forward

    def add_module(self, name, module):
        assert module is None or isinstance(module, Module)
        return super().add_module(name, module)

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
            args = layer.call_inverse(*args, *extras.pop())
            args = pack(layer.process(args, inverse=True))
        return args[0] if len(args) == 1 else args

    def process(self, args, inverse=False):
        if inverse:
            return args
        return super().process(args[:-sum(args[-1]) - 1])

    @property
    def call_function(self):
        return self.function

    @property
    def call_inverse(self):
        return self.inverse

    @property
    def reversible(self):
        return all(layer.reversible for layer in self)

    def reverse(self, mode=None):
        if self.reversed != super().reverse(mode).reversed:
            layers = [layer.reverse() for layer in self]
            for i, layer in enumerate(reversed(layers)):
                self[i] = layer
        return self
