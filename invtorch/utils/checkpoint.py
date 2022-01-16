"""Invertible Checkpoint"""
import functools
import itertools

import torch

from ..autograd.grad_mode import backward_mode, dry_mode, in_dry_mode
from ..random import DelayedRNGFork
from ..utils.tools import pack, requires_grad

__all__ = ['checkpoint']


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

        It doesn't have the contraint of `requires_grad=True` for the inputs
        and the outputs and doesn't set all outputs to be differentiable. It
        also supports invertible functions for extreme memroy savings.
        Refer to https://github.com/xmodar/invtorch for more details.

        Args:
            function: any forward function with positional inputs and outputs
            *args: input positional arguments to be passed to `function`
            seed: same as preserve_rng_state; preserves random number generator
            enabled: disables checkpointing if set to `False`
            inverse: inverse of `function` with positional inputs and outputs
            keep: set of input tensors to keep in memory during inverse mode

        Returns:
            Outputs of `function` with checkpointing if enabled
        """
        if not enabled or in_dry_mode() or not torch.is_grad_enabled():
            return function(*args)
        nonce = torch.tensor((), requires_grad=True)  # force differentiability
        return super().apply(function, inverse, pack(keep), seed, nonce, *args)

    @staticmethod
    def forward(ctx, function, inverse, keep, seed, _, *args):
        ctx.set_materialize_grads(False)

        # capture autocast and RNG states and run the function in dry mode
        ctx.autocast = torch.is_autocast_enabled()
        with dry_mode():
            ctx.forked_rng = DelayedRNGFork.from_tensors(args, seed)
            outputs = function(*args)

        # mark non-differentiable tensors
        output_pack = pack(outputs)
        grad = lambda r: lambda x: torch.is_tensor(x) and r == x.requires_grad
        ctx.mark_non_differentiable(*filter(grad(False), output_pack))
        if not any(map(grad(True), output_pack)):  # non-differentiable outputs
            return outputs

        # get all tensors that should be deallocated from memory `yeet`
        if inverse is None:
            yeet = itertools.repeat(False, len(args))
        else:
            uid = lambda x: x.storage().data_ptr() if torch.is_tensor(x) else 0
            keep = {uid(x) for x in itertools.chain(keep, output_pack, [0])}
            yeet = [uid(x) not in keep for x in args]
            if any(yeet):
                inverse = functools.partial(inverse, *output_pack)
            else:
                inverse = None
        ctx.function, ctx.inverse = function, inverse

        # separate kept and deallocated tensors from other inputs
        ctx.indices, tensors, ctx.deallocated, ctx.inputs = [], [], set(), {}
        for i, (argument, deallocate) in enumerate(zip(args, yeet)):
            if torch.is_tensor(argument):
                ctx.indices.append(i)
                tensors.append(argument)
                if deallocate:
                    argument.storage().resize_(0)  # deallocate the tensor
                    ctx.deallocated.add(i)
            else:
                ctx.inputs[i] = argument
        ctx.save_for_backward(*tensors)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError('.grad()/.backward(inputs=...) not supported')

        # materialize any deallocated tensors by calling inverse
        tensors = ctx.saved_tensors
        if ctx.inverse is not None:
            with torch.inference_mode():
                inverted, ctx.inverse = pack(ctx.inverse()), None
                for i, idx in enumerate(ctx.indices):
                    if idx in ctx.deallocated:
                        tensors[i].set_(inverted[idx])
                del inverted

        # detach input tensors and run function again but in grad_mode
        inputs = ctx.inputs.copy()
        for i, idx in enumerate(ctx.indices):
            inputs[idx] = tensors[i].detach()
            inputs[idx].requires_grad_(tensors[i].requires_grad)
        inputs = [inputs[i] for i in range(len(inputs))]
        with torch.enable_grad(), torch.cuda.amp.autocast(ctx.autocast):
            with backward_mode(), ctx.forked_rng:
                outputs = pack(ctx.function(*inputs))

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
