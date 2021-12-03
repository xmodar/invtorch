"""InvTorch: Basic Invertible Modules https://github.com/xmodar/invtorch"""
import itertools
import functools

import torch
from torch import nn
from torch.nn import functional as F

from .core import checkpoint
from .utils import pack, requires_grad

__all__ = ['Module', 'Linear', 'Conv1d', 'Conv2d', 'Conv3d']


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


class Linear(WrapperModule):
    """Invertible linear module"""
    wrapped_type = nn.Linear

    @functools.wraps(nn.Linear.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(nn.Linear(*args, **kwargs))
        assert self.in_features <= self.out_features, 'few out_features'

    @property
    def reversible(self):
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


class _ConvNd(WrapperModule):
    """Invertible convolution"""
    wrapped_type = nn.modules.conv._ConvNd  # pylint: disable=protected-access

    def __init__(self, module):
        super().__init__(module)
        outputs, inputs = self.flat_weight_shape
        assert inputs <= outputs, f'out_channels/groups={outputs} < {inputs}'
        # TODO: assert kernel_size is still invertible given stride & dilation

    @property
    def reversible(self):
        outputs, inputs = self.flat_weight_shape
        return inputs == outputs, f'out_channels/groups={outputs} != {inputs}'

    # pylint: disable=arguments-differ
    def function(self, inputs, *, strict_forward=False, saved=()):
        if 0 in saved:
            return None
        # TODO: make input padding an opt-in feature
        input_padding = self.get_input_padding(inputs.shape)
        assert sum(input_padding) == 0, f'inputs need padding: {inputs.shape}'
        outputs = self.module.forward(inputs)
        if strict_forward:
            requires_grad(outputs, any=(inputs, self.weight, self.bias))
        return outputs

    def inverse(self, outputs, *, strict_forward=False, saved=()):
        if 0 in saved:
            return None
        old_outputs = outputs
        if self.bias is not None:
            outputs = outputs - self.bias.view(-1, *[1] * self.dim)

        # compute the overlapped inputs
        factor, input_shape = self.flat_weight_shape
        weight = self.weight.view(self.groups, -1, input_shape)
        weight = weight.pinverse().transpose(-1, -2).reshape_as(self.weight)
        inputs = self.conv_transpose(outputs, weight)

        # TODO: make kernel overlaps an opt-in feature
        # normalize the overlapping regions
        one = torch.ones((), device=inputs.device, dtype=inputs.dtype)
        outputs = (one / factor).expand(1, *outputs.shape[1:])
        overlaps_weight = one.expand(self.out_channels, 1, *self.kernel_size)
        overlaps = self.conv_transpose(outputs, overlaps_weight)
        inputs = inputs.div_(overlaps)

        if strict_forward:
            requires_grad(inputs, any=(old_outputs, self.weight, self.bias))
        return inputs

    def get_input_padding(self, input_size):
        """Get the input padding given the input size"""
        def rule(size, kernel_size, stride, padding, dilation):
            margin = 2 * padding - dilation * (kernel_size - 1) - 1
            return size - ((size + margin) // stride) * stride + margin

        input_size = tuple(input_size)[-self.dim:]
        args = self.kernel_size, self.stride, self.padding, self.dilation
        return tuple(map(lambda x: rule(*x), zip(input_size, *args)))

    def conv_transpose(self, inputs, weight):
        """Compute conv_transpose"""
        args = None, self.stride, self.padding, 0, self.groups, self.dilation
        return getattr(F, f'conv_transpose{self.dim}d')(inputs, weight, *args)

    @property
    def dim(self):
        """Number of dimensions"""
        return self.weight.dim() - 2

    @property
    def flat_weight_shape(self):
        """Output and input shapes"""
        shape = self.weight.shape
        return shape[0] // self.groups, shape[1:].numel()


class Conv1d(_ConvNd):
    """Invertible 1D convolution"""
    wrapped_type = nn.Conv1d

    @functools.wraps(nn.Conv1d.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(nn.Conv1d(*args, **kwargs))


class Conv2d(_ConvNd):
    """Invertible 2D convolution"""
    wrapped_type = nn.Conv2d

    @functools.wraps(nn.Conv2d.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(nn.Conv2d(*args, **kwargs))


class Conv3d(_ConvNd):
    """Invertible 3D convolution"""
    wrapped_type = nn.Conv3d

    @functools.wraps(nn.Conv3d.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(nn.Conv3d(*args, **kwargs))
