"""Base Invertible Modules"""
import itertools

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

    @property
    def reversible(self):
        """Whether function and inverse can be switched"""
        return False

    def reverse(self, mode=None):
        """Switch function and inverse"""
        self.reversed = not self.reversed if mode is None else mode
        return self

    def process_inputs(self, inputs):
        """Process the inputs to `self.forward()` as tuples"""
        return inputs

    def forward(self, *inputs, **kwargs):
        """Perform the forward pass"""
        inputs = self.process_inputs(inputs)
        kwargs.setdefault('seed', self.seed)
        use_checkpoint = kwargs.pop('checkpoint', self.checkpoint)
        kwargs['enabled'] = kwargs.get('enabled', True) and use_checkpoint
        function, inverse = self.function, self.inverse
        if self.reversed:
            function, inverse = inverse, function
        if not kwargs.pop('invertible', self.invertible):
            inverse = None
        outputs = checkpoint(function, *inputs, inverse=inverse, **kwargs)
        return self.process_outputs(outputs)

    def process_outputs(self, outputs):
        """Process the outputs of `self.forward()`"""
        return outputs

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
