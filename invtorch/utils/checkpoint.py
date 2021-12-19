"""Invertible Checkpoint"""
import functools

import torch

from ..autograd.grad_mode import dry_mode, in_dry_mode
from ..random import DelayedRNGFork
from ..utils.tools import pack, requires_grad

__all__ = ['checkpoint', 'positional']


def get_tensor_id(inputs, by_storage=True):
    """Get a uniquely identifying key for a tensor based on its storage"""
    assert torch.is_tensor(inputs)
    return inputs.storage().data_ptr() if by_storage else id(inputs)


def get_tensor_id_set(*inputs, by_storage=True):
    """Get a set of only the tensors' ids"""
    get_id = functools.partial(get_tensor_id, by_storage=by_storage)
    return set(map(get_id, filter(torch.is_tensor, inputs)))


def positional(hide_index=None):
    """Decorator for invtorch functions (can hide some outputs)"""
    if callable(hide_index):
        if hasattr(hide_index, 'hide_index'):
            return hide_index
        return positional(None)(hide_index)

    def decorator(wrapped):
        @functools.wraps(wrapped)
        def wrapper(*args):
            args = pack(wrapped(*args))
            assert isinstance(args, tuple), 'outputs must be in a tuple'
            return wrapper.outputs(args)

        def outputs(args):
            args = args[:wrapper.hide_index]
            return args[0] if len(args) == 1 else args

        wrapper.hide_index = hide_index
        wrapper.outputs = outputs
        wrapper.wrapped = wrapped
        return wrapper

    return decorator


class CheckpointFunction(torch.autograd.Function):
    """Improved `torch.utils.checkpoint.CheckpointFunction`"""
    # pylint: disable=abstract-method, arguments-differ, protected-access
    @classmethod
    def apply(
            cls,
            function,
            *args,
            seed=True,
            enabled=True,
            inverse=None,
            keep=(),
    ):
        """Improved `torch.utils.checkpoint.checkpoint`

        Refer to https://github.com/xmodar/invtorch for more details.

        Args:
            function: any forward function wrapped with `invtorch.positional()`
            *args: input arguments `tuple` to be passed to `function`
            seed: same as preserve_rng_state; preserves random number generator
            enabled: disables checkpointing if set to `False`
            inverse: inverse of `function` wrapped with `invtorch.positional()`
            keep: set of input tensors to keep in memory during inverse mode

        Returns:
            Outputs of `function` with checkpointing if enabled
        """
        function = positional(function)
        if not enabled or in_dry_mode() or not torch.is_grad_enabled():
            return function(*args)
        if inverse is not None:
            inverse = positional(inverse)
        nonce = torch.tensor((), requires_grad=True)  # force differentiability
        return super().apply(function, inverse, pack(keep), seed, nonce, *args)

    @staticmethod
    def forward(ctx, function, inverse, keep, seed, _, *args):
        # capture autocast and RNG states and run the function in dry mode
        ctx.set_materialize_grads(False)
        ctx.autocast = torch.is_autocast_enabled()
        with dry_mode():
            ctx.forked_rng = DelayedRNGFork.from_tensors(args, seed)
            outputs = pack(function.wrapped(*args))

        # bookkeep differentiable tensors
        grads = list(map(requires_grad, outputs))
        no_grads = (x for x, g in zip(outputs, grads) if not g)
        ctx.mark_non_differentiable(*filter(torch.is_tensor, no_grads))
        if not any(grads):
            return function.outputs(outputs)  # no output requires gradient

        # get all tensors that should be deallocated from memory
        if inverse is None:
            seep = ()
        else:
            keep = get_tensor_id_set(*keep, *outputs)
            seep = get_tensor_id_set(*args) - keep
            if seep:
                ctx.outputs = outputs
            else:
                inverse = None
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
        return function.outputs(outputs)

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError('.grad()/.backward(inputs=...) not supported')

        # materialize any deallocated tensors by calling inverse
        tensors = ctx.saved_tensors
        if ctx.inverse is not None:
            with torch.inference_mode():
                inverted = pack(ctx.inverse(*ctx.outputs))
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
                outputs = pack(ctx.function.wrapped(*inputs))

        # perform the backward pass on outputs that requires_grad
        outputs_with_grad, args_with_grad = [], []
        for output, arg in zip(outputs, args):
            if arg is not None and requires_grad(output):
                outputs_with_grad.append(output)
                args_with_grad.append(arg)
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = (x.grad if torch.is_tensor(x) else None for x in inputs)
        return (None, None, None, None, None, *grads)


checkpoint = CheckpointFunction.apply
