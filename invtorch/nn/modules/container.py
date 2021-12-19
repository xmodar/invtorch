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

    def function(self, *args, **kwargs):
        cache = kwargs.pop('cache', None)
        if cache is not None:
            all_kwargs = []
            cache['kwargs'], cache = all_kwargs, cache.copy()
            clean_cache = cache
        for layer in self.module:
            if cache is not None:
                cache = clean_cache.copy()
            args = layer.call_function(*pack(args), **kwargs, cache=cache)
            if cache is not None:
                cache.pop(':mode', None)
                all_kwargs.append(cache)
            kwargs = {}
        return args

    def inverse(self, *args, kwargs=None, cache=None):
        # pylint: disable=arguments-differ, unused-argument
        if cache is not None and cache.get(':saved'):
            cache[':saved'] = set()
        for i, layer in enumerate(reversed(self.module), 1):
            rest = {} if kwargs is None else kwargs[-i]
            args = layer.call_inverse(*pack(args), **rest, cache=cache)
        return args

    call_function, call_inverse = function, inverse

    @property
    def reversible(self):
        return all(layer.reversible for layer in self.module)

    def reverse(self, mode=None):
        if self.reversed != super().reverse(mode).reversed:
            layers = reversed(layer.reverse() for layer in self.module)
            self.module = nn.Sequential(*layers)
