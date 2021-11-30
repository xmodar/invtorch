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
    def __init__(self):
        super().__init__()
        self.seed = False  # preserve RNG state in backward
        self.reversed = False  # switch function and inverse
        self.invertible = True  # use inverse if checkpointing is enabled
        self.checkpoint = True  # enables or disables checkpointing

    def function(self, *inputs, strict_forward=False, saved=()):
        """Compute the outputs of the function given the inputs

        The first run of function will be in no_grad mode. Therefore, you must
        manually call `.requires_grad_(True/False)` for all output tensors when
        `strict_forward` is set to `True`. Infer the values from requires_grad
        of `inputs` and `self.parameters()`. You should handle all possible
        combinations or you will get some errors in backward. You can verify
        your implementation by simply calling `self.check_function()`.
        """
        raise NotImplementedError

    def inverse(self, *outputs, strict_forward=False, saved=()):
        """Compute the inputs of the function given the outputs

        Verify your implementation by calling `self.check_inverse()`.
        """
        raise NotImplementedError

    def forward(self, *inputs, **kwargs):
        """Perform the forward pass"""
        kwargs.setdefault('seed', self.seed)
        invertible = kwargs.pop('invertible', self.invertible)
        enabled = kwargs.pop('enabled', True)
        enabled = enabled and kwargs.pop('checkpoint', self.checkpoint)
        return checkpoint(
            self.call_function,
            *inputs,
            enabled=enabled,
            inverse=self.call_inverse if invertible else None,
            **kwargs,
        )

    def reverse(self, mode=None):
        """Switch function and inverse (works if they are fully implemented)"""
        if mode is None:
            mode = not self.reversed
        self.reversed = bool(mode)
        return self

    @property
    def call_function(self):
        """Current function (according to `self.reversed`)"""
        return self.inverse if self.reversed else self.function

    @property
    def call_inverse(self):
        """Current inverse (according to `self.reversed`)"""
        return self.function if self.reversed else self.inverse

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
            'reversed': self.reversed,
            'invertible': self.invertible,
            'checkpoint': self.checkpoint,
        }

    def set_extra_state(self, state):
        self.seed = state['seed']
        self.reversed = state['reversed']
        self.invertible = state['invertible']
        self.checkpoint = state['checkpoint']

    def extra_repr(self):
        return ', '.join(f'{k}={v}' for k, v in self.get_extra_state().items())


class Linear(Module):
    """Invertible linear module"""
    def __init__(self, model):
        super().__init__()
        assert isinstance(model, nn.Linear), f'{type(model)} not nn.Linear'
        self.model = model

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

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
