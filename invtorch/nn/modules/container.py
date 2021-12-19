"""Invertible Container Modules"""
from torch import nn

from ...utils.tools import pack
from .module import WrapperModule

__all__ = ['Sequential']


class Sequential(WrapperModule):
    """Sequential Invertible module"""
    wrapped_type = nn.Sequential

    def __init__(self, *modules):
        super().__init__(nn.Sequential(*modules))

    def function(self, *args):
        extras, counts = [], []
        for layer in self.module:
            layer = layer.call_function
            outputs = pack(layer.wrapped(*args))
            args = pack(layer.outputs(outputs))
            counts.append(len(outputs) - len(args))
            extras.extend(outputs[len(args):])
        self.function.hide_index = len(args)
        return (*args, *extras, counts)

    def inverse(self, *args):
        extras, end = [()] * len(args[-1]), -1
        for i, count in enumerate(reversed(args[end]), 1):
            extras[-i] = args[end - count:end]
            end -= count
        args = args[:end]
        for layer in reversed(self.module):
            args = pack(layer.call_inverse(*args, *extras.pop()))
        return args[0] if len(args) == 1 else args

    @property
    def call_function(self):
        return self.function

    @property
    def call_inverse(self):
        return self.inverse

    @property
    def reversible(self):
        return all(layer.reversible for layer in self.module)

    def reverse(self, mode=None):
        if self.reversed != super().reverse(mode).reversed:
            layers = reversed([layer.reverse() for layer in self.module])
            self.module = nn.Sequential(*layers)
        return self
