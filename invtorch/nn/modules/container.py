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

    def function(self, *inputs, strict=None):
        # pylint: disable=arguments-differ
        extras, counts = [], []
        for layer in self.module:
            outputs = pack(layer.call_function(*inputs, strict=strict))
            inputs = pack(layer.process_outputs(*outputs))
            counts.append(len(outputs) - len(inputs) if strict else 0)
            if strict:
                extras.extend(outputs[len(inputs):])
        return (*inputs, *extras, counts)

    def inverse(self, *outputs, saved=()):
        # pylint: disable=arguments-differ
        extras, end = [()] * len(outputs[-1]), -1
        for i, count in enumerate(reversed(outputs[end]), 1):
            extras[-i] = outputs[end - count:end]
            end -= count
        outputs = outputs[:end]
        kwargs = {}
        for layer in reversed(self.module):
            outputs = outputs + extras.pop()
            if not extras and saved:
                kwargs['saved'] = saved
            outputs = pack(layer.call_inverse(*outputs, **kwargs))
        return outputs[0] if len(outputs) == 1 else outputs

    def process_outputs(self, *outputs):
        return super().process_outputs(*outputs[:-sum(outputs[-1]) - 1])

    call_function, call_inverse = function, inverse

    @property
    def reversible(self):
        return all(layer.reversible for layer in self.module)

    def reverse(self, mode=None):
        if self.reversed != super().reverse(mode).reversed:
            layers = reversed(layer.reverse() for layer in self.module)
            self.module = nn.Sequential(*layers)
