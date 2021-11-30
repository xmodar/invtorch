"""InvTorch: Basic Invertible Modules https://github.com/xmodar/invtorch"""
import itertools

import torch
from torch import nn
from torch.nn import functional as F

from .core import checkpoint
from .utils import pack, requires_grad

__all__ = ['Module', 'Linear']


class Module(nn.Module):
    """Base invertible module `inputs = self.inverse(*self.function(*inputs))`

    Use this with great caution. Refer to the notes in `invtorch.checkpoint()`
    Source: https://github.com/xmodar/invtorch
    """
    reversible = False  # whether function and inverse can be switched

    def __init__(self):
        super().__init__()
        self.seed = False  # preserve RNG state in backward
        self.checkpoint = True  # enables or disables checkpointing
        self.invertible = True  # use inverse if checkpointing is enabled
        self.reversed = False  # switch function and inverse

    def function(self, *inputs, strict_forward=False, saved=()):
        """Compute the outputs of the function given the inputs

        The first run of function will be in no_grad mode. Therefore, you must
        manually call `.requires_grad_(True/False)` for all output tensors when
        `strict_forward` is set to `True`. Infer the values from requires_grad
        of `inputs` and all used parameters. You should handle all possible
        combinations or you will get some errors in backward. You can verify
        your implementation by simply calling `self.check_function()`.
        """
        raise NotImplementedError

    @property
    def call_function(self):
        """Current function (according to `self.reversed`)"""
        return self.inverse if self.reversed else self.function

    def inverse(self, *outputs, strict_forward=False, saved=()):
        """Compute the inputs of the function given the outputs

        Verify your implementation by calling `self.check_inverse()`.
        """
        raise NotImplementedError

    @property
    def call_inverse(self):
        """Current inverse (according to `self.reversed`)"""
        return self.function if self.reversed else self.inverse

    def reverse(self, mode=None):
        """Switch function and inverse"""
        self.reversed = not self.reversed if mode is None else mode
        return self

    def forward(self, *inputs, **kwargs):
        """Perform the forward pass"""
        self.process_checkpoint_arguments(kwargs)
        return checkpoint(kwargs.pop('function'), *inputs, **kwargs)

    def process_checkpoint_arguments(self, kwargs):
        """Populate the keyword arguments of `invtorch.checkpoint()`"""
        kwargs.setdefault('seed', self.seed)
        use_checkpoint = kwargs.pop('checkpoint', self.checkpoint)
        kwargs['enabled'] = kwargs.get('enabled', True) and use_checkpoint
        function = kwargs.pop('function', self.function)
        inverse = kwargs.pop('inverse', self.inverse)
        if kwargs.pop('reversed', self.reversed):
            function, inverse = inverse, function
        if not kwargs.pop('invertible', self.invertible):
            inverse = None
        kwargs['function'], kwargs['inverse'] = function, inverse
        return kwargs

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
        if value:
            assert self.reversible, 'module is not reversible'
            self._reversed = True
        else:
            self._reversed = False

    def check_function(self, *inputs):
        """Check if `self.call_function()` is consistent when strict_forward"""
        with torch.enable_grad():
            outputs1 = pack(self.call_function(*inputs, strict_forward=False))
        with torch.no_grad():
            outputs2 = pack(self.call_function(*inputs, strict_forward=True))
        assert len(outputs2) == len(outputs1), 'number of outputs'
        grads1 = list(map(requires_grad, outputs1))
        grads2 = list(map(requires_grad, outputs2))
        bad = {i for i, (g1, g2) in enumerate(zip(grads1, grads2)) if g1 != g2}
        expected = [(i in bad) != g for i, g in enumerate(grads2)]
        assert not bad, f'Received: {grads2}\nExpected: {expected}'
        return True

    @torch.inference_mode()
    def check_inverse(self, *inputs, atol=1e-5, rtol=1e-3):
        """Check if `self.call_inverse()` computes correct input tensors"""
        outputs = pack(self.call_inverse(*pack(self.call_function(*inputs))))
        for inputs, outputs in itertools.zip_longest(inputs, outputs):
            is_tensor = torch.is_tensor(inputs)
            assert is_tensor == torch.is_tensor(outputs)
            assert not is_tensor or torch.allclose(inputs, outputs, rtol, atol)
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

    def extra_repr(self):
        extra = f'reversed={self.reversed}, checkpoint={self.checkpoint}'
        if self.checkpoint:
            extra += f', invertible={self.invertible}, seed={self.seed}'
        return extra


class WrapperModule(Module):
    """Base wrapper invertible module"""
    # pylint: disable=abstract-method
    wrapped_types = ()

    def __init__(self, module):
        assert self.wrapped_types, 'define wrapped_types'
        assert isinstance(module, self.wrapped_types), (
            f'{type(module).__name__} is not in <{self.wrapped_types}>')
        super().__init__()
        self.module = module

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Linear(WrapperModule):
    """Invertible linear module"""
    wrapped_types = nn.Linear

    def __init__(self, module):
        super().__init__(module)
        assert self.out_features >= self.in_features, 'few out_features'

    @property
    def reversible(self):
        """Whether function and inverse can be switched"""
        return self.in_features == self.out_features

    # pylint: disable=arguments-differ
    def function(self, inputs, *, strict_forward=False, saved=()):
        """Compute the outputs of the function"""
        if 0 in saved:
            return None
        outputs = F.linear(inputs, self.weight, self.bias)
        if strict_forward:
            requires_grad(outputs, any=(inputs, self.weight, self.bias))
        return outputs

    def inverse(self, outputs, *, strict_forward=False, saved=()):
        """Compute the inputs of the function"""
        if 0 in saved:
            return None
        if self.bias is not None:
            outputs = outputs - self.bias
        inputs = F.linear(outputs, self.weight.pinverse())
        if strict_forward:
            requires_grad(inputs, any=(outputs, self.weight, self.bias))
        return inputs
