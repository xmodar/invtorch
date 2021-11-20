"""InvTorch: Core Invertible Utilities https://github.com/xmodar/invtorch"""
import itertools

import torch
from torch.utils.checkpoint import checkpoint as _checkpoint

from .utils import (DelayedRNGFork, get_tensor_id, get_tensor_id_set, pack,
                    requires_grad)

__all__ = ['InvertibleModule', 'checkpoint']


def checkpoint(
        function,
        *inputs,
        seed=True,
        strict=False,
        enabled=True,
        inverse=None,
        keep=(),
):
    """Same as `torch.utils.checkpoint.checkpoint` with extra functionalities

    The original checkpoint needs at least one input with `requires_grad` set
    to `True` to work. This is fine as long as the `function` doesn't have any
    tensors that require gradients. In such case, no gradients will be computed
    and PyTorch will raise a UserWarning. The checkpoint will be disabled and
    the code will run in `torch.no_grad()` mode.

    Moreover, by default all output tensors will have `requires_grad` set
    to `True` even if they shouldn't. This is because, in the forward pass,
    the `function` will be called in `no_grad` mode and it is difficult to
    cheaply keep track of which output will require gradient beforehand.
    Refer to https://pytorch.org/docs/1.10.0/checkpoint.html for more details.

    Instead, this function has two new flags; `strict` and `enabled`.
    Setting `enabled` to `False`, will disable all checkpointing logic. Else,
    when `strict` is set to `False`, it will use the original checkpoint as is.
    Running in `strict` mode incurs two main changes to the original behavior:
        (1) When no input tensor requires gradient but `function` generates
            outputs that do require gradients, the checkpoint will still work
        (2) All outputs will have `requires_grad` set to `False` by default.
            To specify what tensors actually require gradients, `function` will
            expect a keyword argument `strict_forward` which will be `True` if
            we cannot automatically track this information. Here is an example:
    ```python
    def function(x, y, strict_forward=False):
        z = x + y
        if strict_forward:  # set requires_grad for all outputs
            z.requires_grad = x.requires_grad or y.requires_grad
            # no need to set for y as it is already set
        return z, y
    ```
           Debug your code carefully and try to cover all the cases. Don't
           forget to account for any used parameters that requires_grad.

    In addition, this function allows a more extreme version of checkpointing.
    If `function` was invertible with respect to its input tensors, then we can
    deallocate them as well and recover them using `inverse`. It should be a
    function that takes all output arguments of `function` and returns all the
    computed inputs. In general, `inverse` doesn't need to be differentiable
    and it will always run in `torch.inference_mode()` mode with `strict=True`.
    It shouldn't have side effects and must not modify the outputs (its inputs)
    in-place in most cases. It only needs to compute the input tensors and can
    return anything for non-tensor inputs but it is a good practice to return
    them to allow for nesting invertible functions.

    For more granular control, it also expects an additional keyword argument
    `saved` which is the set of the positions of the inputs that are in memory.
    Here is the inverse of the previous example:
    ```python
    def inverse(z, y, saved=()):
        x = z - y if 0 in saved else None
        return x, y
    ```
    You might notice that `1` (for `y`) will always be in `saved` since it is
    an output of `function` and might be used in later operations. This was
    detected automatically by checking if any output share the same memory with
    an input. To manually select tensors to keep in memory, you can pass them
    to the `keep` argument as a tuple of tensors that are in the input already.

    There are few caveats to consider though. Invertible functions are hard to
    define without requiring more memory. Moreover, they are prone to numerical
    instabilities (e.g., multiplying by numbers close to zero). Even if we can
    get away with these fundamental problems, there are technical details to
    consider here. There is no way of guarding against accessing the data in
    the input tensors after calling the function and before the backward pass.
    It is up to the user to ensure this. Otherwise, it is possible to run into
    illegal memory access errors. Think of residual connections as an example.
    In `x + f(x)`, assuming `f` is an invertible checkpoint, `x` will be freed
    from memory before the sum is computed. On the other hand, we can maybe
    use `x.clone() + f(x)` (not `f(x) + x.clone()`!) but now we have a copy of
    `x` in memory. It is recommended to encapsulate this inside `f` itself or
    use the simple checkpoint instead. Other alternatives exists and you should
    study your case carefully before deciding to use this. Fore instance, check
    out `torch.autograd.graph.saved_tensors_hooks()` and `graph.save_on_cpu()`.
    Source: https://github.com/xmodar/invtorch

    Args:
        function: this is any forward function with positional arguments
        *inputs: input arguments tuple to be passed to `function`
        seed: same as preserve_rng_state; preserves the random number generator
        strict: `requires_grad` for outputs is set manually in `function`
        enabled: disables checkpointing if set to False
        inverse: inverse of `function` to deallocate the inputs as well
        keep: set of input tensors to keep in memory during inverse mode

    Returns:
        Outputs of `function(*inputs)` with checkpointing if required
    """
    if not enabled or not torch.is_grad_enabled():  # no checkpointing
        return function(*inputs)

    # find out which mode to run? nonstrict, strict, or strict inverse
    strict = strict or inverse is not None  # inverse is always strict

    if not strict:  # use torch.utils.checkpoint.checkpoint
        return _checkpoint(function, *inputs, preserve_rng_state=seed)

    return CheckpointFunction.apply(function, inverse, keep, seed, *inputs)


class CheckpointFunction(torch.autograd.Function):
    """Improved `torch.utils.checkpoint.CheckpointFunction` (invertible)"""
    # pylint: disable=abstract-method, arguments-differ, protected-access
    @classmethod
    def apply(cls, function, inverse, keep, seed, *inputs):
        """Refer to `invtorch.checkpoint()` for the arguments' details"""
        dif = torch.tensor((), requires_grad=True)  # ensures differentiability
        return super().apply(function, inverse, pack(keep), seed, dif, *inputs)

    @staticmethod
    def forward(ctx, function, inverse, keep, seed, _, *args):
        # capture autocast and RNG states and run the function
        ctx.set_materialize_grads(False)
        ctx.autocast = torch.is_autocast_enabled()
        devices = (x.device for x in args if torch.is_tensor(x) and x.is_cuda)
        ctx.forked_rng = DelayedRNGFork(devices, seed)
        outputs = function(*args, strict_forward=True)
        assert isinstance(outputs, (torch.Tensor, tuple))

        # bookkeep differentiable tensors
        ctx.grads, non_differentiable = [], []
        for argument in pack(outputs):
            does_require = False
            if torch.is_tensor(argument):
                if argument.requires_grad:
                    does_require = True
                else:
                    non_differentiable.append(argument)
            ctx.grads.append(does_require)
        ctx.mark_non_differentiable(*non_differentiable)
        if not any(ctx.grads):
            return outputs  # apparently, `function` was not differentiable

        # get all tensors that should be deallocated from memory
        if inverse is None:
            seep = ()
        else:
            keep = get_tensor_id_set(*keep, *pack(outputs))
            seep = get_tensor_id_set(*args) - keep
            inverse = inverse if seep else None
        ctx.function, ctx.inverse = function, inverse

        # separate kept and deallocated tensors from other inputs
        ctx.indices, tensors, ctx.deallocated, ctx.inputs = [], [], set(), {}
        for i, argument in enumerate(args):
            if torch.is_tensor(argument):
                ctx.indices.append(i)
                tensors.append(argument)
                if get_tensor_id(argument) in seep:
                    argument.storage().resize_(0)  # deallocate the tensor
                    ctx.deallocated.add(i)
            else:
                ctx.inputs[i] = argument
        ctx.save_for_backward(*tensors)
        if seep:
            ctx.outputs = outputs
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError('.grad()/.backward(inputs=...) not supported')

        # materialize any deallocated tensors by calling inverse
        tensors = ctx.saved_tensors
        if ctx.inverse is not None:
            with torch.inference_mode():
                saved = set(range(len(ctx.grads))) - ctx.deallocated
                inverted = pack(ctx.inverse(*pack(ctx.outputs), saved=saved))
                for i in ctx.deallocated:
                    tensors[i].set_(inverted[i])
            ctx.inverse = ctx.outputs = inverted = None

        # detach input tensors and run function again but in grad_mode
        inputs = ctx.inputs.copy()
        for i, idx in enumerate(ctx.indices):
            inputs[idx] = tensors[i].detach()
            inputs[idx].requires_grad_(tensors[i].requires_grad)
        inputs = [inputs[i] for i in range(len(inputs))]
        with ctx.forked_rng:
            with torch.enable_grad(), torch.cuda.amp.autocast(ctx.autocast):
                outputs = pack(ctx.function(*inputs, strict_forward=False))

        # check if requires_grad matches that of strict_forward mode
        current = map(requires_grad, outputs)
        bad = {i for i, (g, c) in enumerate(zip(ctx.grads, current)) if g != c}
        if bad:
            expected = [(i in bad) != g for i, g in enumerate(ctx.grads)]
            msg = ('manually set requires_grad for output tensors in '
                   'strict mode mismatched automatically set values in '
                   'the backward pass. Please, debug your implementation '
                   'carfully and try to cover all the cases. Keep in mind '
                   'the paramters of any model you call in `function`.'
                   f'\nExpected: {expected}\nReceived: {ctx.grads}')
            raise RuntimeError(msg)

        # perform the backward pass on outputs that requires_grad
        outputs_with_grad, args_with_grad = [], []
        for output, arg in zip(outputs, args):
            if arg is not None and requires_grad(output):
                outputs_with_grad.append(output)
                args_with_grad.append(arg)
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = (x.grad if torch.is_tensor(x) else None for x in inputs)
        return (None, None, None, None, None, *grads)


class InvertibleModule(torch.nn.Module):
    """Base invertible `inputs = self.inverse(*self.function(*inputs))`

    Use this with great caution. Refer to the notes in `invtorch.checkpoint()`
    Source: https://github.com/xmodar/invtorch
    """
    keep = ()  # input indices to keep in memory; by default, keep nothing

    def __init__(self, invertible=True, checkpoint=True, seed=False):
        # pylint: disable=redefined-outer-name
        super().__init__()
        self.seed = bool(seed)  # preserve RNG state in backward
        self.invertible = invertible  # use inverse if checkpointing is enabled
        self.checkpoint = checkpoint  # enables or disables checkpointing

    def forward(self, *inputs):
        """Perform the forward pass"""
        disabled = (not self.checkpoint or not torch.is_grad_enabled()
                    or not (requires_grad(any=self.parameters())
                            or requires_grad(any=inputs)))
        return checkpoint(
            self.function,
            *inputs,
            seed=self.seed,
            strict=True,
            enabled=not disabled,
            inverse=self.inverse if self.invertible else None,
            keep=tuple(inputs[i] for i in self.keep),
        )

    def function(self, *inputs, strict_forward=False):
        """Compute the outputs of the function given the inputs

        The first run of function will be in no_grad mode. Thus, you must
        manually call `.requires_grad_(True/False)` for all output tensors when
        `strict_forward` is set to `True`. Infer the values from requires_grad
        of `inputs` and `self.parameters()`. You should handle all possible
        combinations or you will get some errors in backward. You can verify
        your implementation by simply calling `self.check_function()`.
        """
        raise NotImplementedError

    def inverse(self, *outputs, saved=()):
        """Compute the inputs of the function given the outputs

        Verify your implementation by calling `self.check_inverse()`.
        """
        raise NotImplementedError

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
        """Check if `self.function()` is consistent in strict_forward mode"""
        with torch.enable_grad():
            outputs = pack(self.function(*inputs, strict_forward=False))
        with torch.no_grad():
            dry_outputs = pack(self.function(*inputs, strict_forward=True))
        assert len(dry_outputs) == len(outputs), 'number of outputs'
        grads = map(requires_grad, outputs)
        dry = map(requires_grad, dry_outputs)
        bad = {i for i, (d, g) in enumerate(zip(dry, grads)) if d != g}
        expected = [(i in bad) != g for i, g in enumerate(dry)]
        assert not bad, f'Received: {dry}\nExpected: {expected}'
        return True

    @torch.inference_mode()
    def check_inverse(self, *inputs, atol=1e-5, rtol=1e-3):
        """Check if `self.inverse()` computes correct input tensors"""
        outputs = pack(self.inverse(*pack(self.function(*inputs))))
        for inputs, outputs in itertools.zip_longest(inputs, outputs):
            is_tensor = torch.is_tensor(inputs)
            assert is_tensor == torch.is_tensor(outputs)
            assert not is_tensor or torch.allclose(inputs, outputs, rtol, atol)
        return True
