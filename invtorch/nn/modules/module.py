"""Base Invertible Modules"""
import itertools
from contextlib import contextmanager

import torch
from torch import nn

from ...autograd.grad_mode import backward_mode, dry_mode
from ...utils.checkpoint import checkpoint
from ...utils.tools import pack

__all__ = ['Module']


class Module(nn.Module):
    """Base invertible module"""
    def __init__(self):
        super().__init__()
        self.seed = False  # preserve RNG state in backward
        self.checkpoint = True  # enables or disables checkpointing
        self.invertible = True  # use inverse if checkpointing is enabled
        self._reversed = False  # switch function and inverse

    def forward(self, *args, **kwargs):
        """Perform the forward pass"""
        private = {
            'seed': self.seed,
            'enabled': self.checkpoint,
            'inverse': self.call_inverse if self.invertible else None,
        }
        assert all(k not in kwargs for k in private), 'got an illegal argument'
        kwargs.update(private)
        return self.process(checkpoint(self.call_function, *args, **kwargs))

    def function(self, *args):
        """Compute the outputs of the function given the inputs"""
        raise NotImplementedError

    def inverse(self, *args):
        """Compute the inputs of the function given the outputs"""
        raise NotImplementedError

    @property
    def call_function(self):
        """Current function (according to `self.reversed`)"""
        return self.inverse if self.reversed else self.function

    @property
    def call_inverse(self):
        """Current inverse (according to `self.reversed`)"""
        return self.function if self.reversed else self.inverse

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

    @property
    def num_outputs(self):
        """End index to slice `call_function()`'s outputs in `forward()`"""
        return None

    @property
    def num_inputs(self):
        """End index to slice `call_inverse()`'s outputs in `forward()`"""
        return None

    def process(self, args, inverse=False):
        """Process the outputs of `call_function()` or `call_inverse()`"""
        args = pack(args)
        assert isinstance(args, tuple), 'should only output a `tuple`'
        num_args = self.num_inputs if inverse else self.num_outputs
        if num_args is None:
            num_args = len(args)
        elif num_args < 0:
            num_args += len(args)
        assert 0 < num_args <= len(args), f'needs {num_args} args'
        return args[0] if num_args == 1 else args[:num_args]

    def check(self, *args, rtol=1e-3, atol=1e-5):
        """Check invertability and second forward pass consistency"""
        def check(args1, args2, message):
            for arg1, arg2 in itertools.zip_longest(args1, args2):
                is_tensor = torch.is_tensor(arg1)
                assert is_tensor == torch.is_tensor(arg2), message
                same = not is_tensor or torch.allclose(arg1, arg2, rtol, atol)
                assert same, message

        with dry_mode():
            outputs = pack(self.call_function(*args))
        with torch.inference_mode():
            inputs = pack(self.call_inverse(*outputs))
        check(args, inputs, 'inverted tensors mismatch (try double precision)')
        with backward_mode():
            second = pack(self.call_function(*args))
        if self.seed:
            message = 'second forward pass mismatched despite `self.seed=True`'
        else:
            message = 'second forward pass mismatched (try `self.seed=True`)'
        check(outputs, second, message)
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

    def __repr__(self):
        return 'Inv' + super().__repr__()
