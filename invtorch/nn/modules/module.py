"""Base Invertible Modules"""
import itertools
from contextlib import contextmanager

import torch
from torch import nn

from ...utils.checkpoint import checkpoint
from ...utils.tools import pack, requires_grad

__all__ = ['Module', 'WrapperModule']


class Module(nn.Module):
    """Base invertible module `inputs = self.inverse(*self.function(*inputs))`

    Use this with great caution. Refer to the notes in `invtorch.checkpoint()`
    Source: https://github.com/xmodar/invtorch
    """
    def __init__(self):
        super().__init__()
        self.seed = False  # preserve RNG state in backward
        self.checkpoint = True  # enables or disables checkpointing
        self.invertible = True  # use inverse if checkpointing is enabled
        self._reversed = False  # switch function and inverse

    def function(self, *inputs, strict=None, saved=()):
        """Compute the outputs of the function given the inputs"""
        raise NotImplementedError

    @property
    def call_function(self):
        """Current function (according to `self.reversed`)"""
        return self.inverse if self.reversed else self.function

    def inverse(self, *outputs, strict=None, saved=()):
        """Compute the inputs of the function given the outputs"""
        raise NotImplementedError

    @property
    def call_inverse(self):
        """Current inverse (according to `self.reversed`)"""
        return self.function if self.reversed else self.inverse

    num_function_outputs = num_inverse_outputs = None

    @property
    def num_outputs(self):
        """Current number of outputs (according to `self.reversed`)"""
        if self.reversed:
            return self.num_inverse_outputs
        return self.num_function_outputs

    @property
    def reversible(self):
        """Whether function and inverse can be switched"""
        return False

    def reverse(self, mode=None):
        """Switch function and inverse"""
        if not self.reversed if mode is None else mode:
            assert self.reversible, 'module is not reversible'
            self._reversed = True
        else:
            self._reversed = False
        return self

    def forward(self, *inputs, **kwargs):
        """Perform the forward pass"""
        private = {
            'seed': self.seed,
            'enabled': self.checkpoint,
            'inverse': self.call_inverse if self.invertible else None,
        }
        assert all(k not in kwargs for k in private), 'got an illegal argument'
        kwargs.setdefault('strict', True)
        kwargs.update(private)
        outputs = checkpoint(self.call_function, *inputs, **kwargs)
        return self.process_outputs(*pack(outputs))

    def process_outputs(self, *outputs):
        """Get only `self.forward()` outputs"""
        num_outputs = self.num_outputs
        if num_outputs is None:
            num_outputs = len(outputs)
        elif num_outputs < 0:
            num_outputs += len(outputs)
        assert 0 < num_outputs <= len(outputs), f'need {num_outputs} outputs'
        assert not requires_grad(any=outputs[num_outputs:]), (
            'discarded outputs must not be differentiable')
        return outputs[0] if num_outputs == 1 else outputs[:num_outputs]

    @property
    def checkpoint(self):
        """Whether the module is in checkpoint or pass_through mode"""
        return self._checkpoint

    @checkpoint.setter
    def checkpoint(self, value):
        if value:
            self._checkpoint = True
        else:
            self._checkpoint = self._invertible = False

    @property
    def invertible(self):
        """Whether the module is in invertible or simple checkpoint mode"""
        return self._checkpoint and self._invertible

    @invertible.setter
    def invertible(self, value):
        if value:
            self._invertible = self._checkpoint = True
        else:
            self._invertible = False

    @property
    def reversed(self):
        """Whether function and inverse should be switched"""
        return self._reversed

    @reversed.setter
    def reversed(self, value):
        self.reverse(value)

    def check_function(self, *inputs):
        """Check if `self.call_function()` is consistent when `strict=True`"""
        with torch.enable_grad():
            outputs1 = pack(self.call_function(*inputs, strict=False))
            outputs1 = pack(self.process_outputs(*outputs1))
        with torch.no_grad():
            outputs2 = pack(self.call_function(*inputs, strict=True))
            outputs2 = pack(self.process_outputs(*outputs2))
        assert len(outputs2) == len(outputs1), 'number of outputs'
        grads1 = list(map(requires_grad, outputs1))
        grads2 = list(map(requires_grad, outputs2))
        bad = {i for i, (g1, g2) in enumerate(zip(grads1, grads2)) if g1 != g2}
        expected = [(i in bad) != g for i, g in enumerate(grads2)]
        assert not bad, f'Received: {grads2}\nExpected: {expected}'
        return True

    @torch.no_grad()
    def check_inverse(self, *inputs, atol=1e-5, rtol=1e-3):
        """Check if `self.call_inverse()` computes correct input tensors"""
        outputs = pack(self.call_function(*inputs, strict=True))
        outputs = pack(self.call_inverse(*outputs))
        for inputs, outputs in itertools.zip_longest(inputs, outputs):
            is_tensor = torch.is_tensor(inputs)
            assert is_tensor == torch.is_tensor(outputs), 'out types mismatch'
            same = not is_tensor or torch.allclose(inputs, outputs, rtol, atol)
            assert same, 'an inverted tensor mismatched (try double precision)'
        return True

    def get_extra_state(self):
        return {
            'seed': self.seed,
            'checkpoint': self.checkpoint,
            'invertible': self.invertible,
            'reversed': self.reversed,
        }

    def set_extra_state(self, state):
        self.seed = state['seed']
        self.checkpoint = state['checkpoint']
        self.invertible = state['invertible']
        self.reversed = state['reversed']

    @contextmanager
    def temp_mode(self, **kwargs):
        """Set, temporarily, the mode of the model"""
        state = {}
        for key in ('seed', 'checkpoint', 'invertible', 'reversed'):
            state[key] = getattr(self, key)
            if key in kwargs and state[key] == bool(kwargs[key]):
                kwargs.pop(key)
        assert all(k in state for k in kwargs), 'got an illegal argument'
        if 'checkpoint' in kwargs and 'invertible' in kwargs:
            assert kwargs['checkpoint'] or not kwargs['invertible'], (
                'set either `checkpoint` or `invertible` or avoid conflict')
        try:
            for key, value in kwargs.items():
                setattr(self, key, value)
            yield self
        finally:
            for key, value in state.items():
                setattr(self, key, value)

    def extra_repr(self):
        extra = f'reversed={self.reversed}, checkpoint={self.checkpoint}'
        if self.checkpoint:
            extra += f', invertible={self.invertible}, seed={self.seed}'
        return extra


class WrapperModule(Module):
    """Base wrapper invertible module"""
    # pylint: disable=abstract-method
    wrapped_type = ()

    def __init__(self, module):
        assert self.wrapped_type, 'define wrapped_type'
        assert isinstance(module, self.wrapped_type), (
            f'{type(module).__name__} is not in <{self.wrapped_type}>')
        super().__init__()
        self.module = module

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
