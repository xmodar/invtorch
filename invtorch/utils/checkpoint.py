"""Invertible Checkpoint"""
import functools

import torch
from torch.utils.checkpoint import checkpoint as _checkpoint

from ..random import DelayedRNGFork
from ..utils.tools import pack, requires_grad

__all__ = ['checkpoint']


def get_tensor_id(inputs, by_storage=True):
    """Get a uniquely identifying key for a tensor based on its storage"""
    assert torch.is_tensor(inputs)
    return inputs.storage().data_ptr() if by_storage else id(inputs)


def get_tensor_id_set(*inputs, by_storage=True):
    """Get a set of only the tensors' ids"""
    get_id = functools.partial(get_tensor_id, by_storage=by_storage)
    return set(map(get_id, filter(torch.is_tensor, inputs)))


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
    to `True` to work. This is fine if `function` doesn't output tensors that
    require gradients due to some hidden parameters with `requires_grad=True`.
    In such case, a UserWarning will be raised and the code will run as if it
    was run in `torch.no_grad()` mode.

    Moreover, by default all output tensors will have `requires_grad=True` even
    if they shouldn't. This is because, in the forward pass, `function` will be
    called in `no_grad` mode and it is difficult to cheaply keep track of which
    output will require gradients beforehand.
    Refer to https://pytorch.org/docs/1.10.0/checkpoint.html for more details.

    Instead, this function has two new flags; `strict` and `enabled`.
    Setting `enabled` to `False`, will disable all checkpointing logic.
    However, when `strict=False`, it will use the original checkpoint as is.
    Else, when `strict=True`, the original behavior will change in two ways:
        (1) When no input tensor requires gradient but `function` generates
            outputs that do require gradients, the checkpoint will still work
        (2) All outputs will have `requires_grad` set to `False` by default.
            To specify what tensors actually require gradients, `function` will
            expect a keyword argument `strict` which will be `True` if we
            cannot automatically track this information. Here is an example:
    ```python
    def function(x, y, strict=None):
        z = x + y
        if strict:  # set requires_grad for all outputs
            z.requires_grad = x.requires_grad or y.requires_grad
            # no need to set for y as it is already set
        return z, y
    ```
           Debug your code carefully and try to cover all the cases. Don't
           forget to account for any used parameters that requires_grad.

    In addition, this function allows a more extreme version of checkpointing.
    If `function` was invertible with respect to its input tensors, then we can
    deallocate and recover them later using `inverse`. It should be a function
    that takes `function`'s outputs and returns computed `function`'s inputs.
    In general, `inverse` doesn't need to be differentiable and it'll run in
    `torch.inference_mode()`. It shouldn't have side effects and mustn't modify
    the outputs (its inputs) in-place in most cases. It only needs to compute
    the input tensors and can return anything for non-tensor inputs but it is
    a good practice to return them to allow for nesting invertible functions.

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
        outputs = function(*args, strict=True)
        assert isinstance(outputs, (torch.Tensor, tuple))

        # bookkeep differentiable tensors
        ctx.grads = list(map(requires_grad, pack(outputs)))
        no_grads = (x for x, g in zip(pack(outputs), ctx.grads) if not g)
        ctx.mark_non_differentiable(*filter(torch.is_tensor, no_grads))
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
            saved = set(ctx.indices) - ctx.deallocated
            kwargs = dict(saved=saved) if saved else {}
            with torch.inference_mode():
                inverted = pack(ctx.inverse(*pack(ctx.outputs), **kwargs))
                for i, idx in enumerate(ctx.indices):
                    if idx in ctx.deallocated:
                        tensors[i].set_(inverted[idx])
            ctx.inverse = ctx.outputs = inverted = None

        # detach input tensors and run function again but in grad_mode
        inputs = ctx.inputs.copy()
        for i, idx in enumerate(ctx.indices):
            inputs[idx] = tensors[i].detach()
            inputs[idx].requires_grad_(tensors[i].requires_grad)
        inputs = [inputs[i] for i in range(len(inputs))]
        with torch.enable_grad(), torch.cuda.amp.autocast(ctx.autocast):
            with ctx.forked_rng:
                outputs = pack(ctx.function(*inputs, strict=False))

        # check if requires_grad matches that of function call in forward
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
