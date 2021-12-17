"""Invertible Checkpoint"""
import functools
import itertools

import torch

from ..autograd.grad_mode import dry_mode, in_dry_mode
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


def flat(function):
    """Change a function to input and output only positional arguments"""
    @functools.wraps(function)
    def function_(cache, pure, keys, *args):
        args, kwargs = args[len(keys):], args[:len(keys)]
        kwargs = {k: v for k, v in zip(keys, kwargs) if not k.startswith(':')}
        outs = function(*args, **kwargs, cache=cache)
        assert isinstance(outs, (torch.Tensor, tuple)), type(outs).__name__
        cache[':single'] = torch.is_tensor(outs)
        if pure:
            for key in list(cache.keys()):
                if key.startswith(':'):
                    cache.pop(key)
        return tuple(flat.args(pack(outs), cache))

    return function_


# transform args and kwargs to only positional arguments for a flat function
flat.args = lambda a, k: itertools.chain([tuple(k.keys())], k.values(), a)

# get keyword argument value give the key from the output of a flat function
flat.get = lambda o, k: o[o[0].index(k) + 1]

# extract only the positional arguments from the output of a flat function
flat.outs = lambda o: o[-1] if flat.get(o, ':single') else o[len(o[0]) + 1:]


class CheckpointFunction(torch.autograd.Function):
    """Improved `torch.utils.checkpoint.CheckpointFunction` (invertible)"""
    # pylint: disable=abstract-method, arguments-differ, protected-access
    @classmethod
    def apply(cls, function, inverse, keep, seed, enabled, /, *args, **kwargs):
        """Improved `torch.utils.checkpoint.checkpoint`

        The original checkpoint cannot track which tensors `requires_grad`.
        See https://pytorch.org/docs/1.10.0/checkpoint.html for more details.

        Instead, this checkpoint doesn't have this issue and it also supports
        invertible functions which will allow deallocating some input tensors,
        to save more memory, that will be recomputed in the backward pass.

        Example of a function and its inverse and their structure. Notice, how
        they both need to accept an optional keyword argument `cache`:
        ```python
        def function(x, constant=2, cache=None):
            assert constant != 0, 'not invertible if `constant` is zero'
            if cache is not None:  # save needed values for the inverse
                cache['constant'] = constant
            return x * constant


        def inverse(x, constant=2, cache=None):
            return x / constant


        # the cache is used to pass keyword arguments to inverse
        # make sure to always detach any tensor in the cache
        cache = {}
        y = function(x, 5, cache)
        x = inverse(x, **cache)
        ```

        When using this checkpoint with invertible inputs, `function` will be
        called once in the forward pass with `cache[':mode'] == 'forward'`.
        Then, later in the backward pass, `inverse` is called followed by a
        call to `function` again with `cache[':mode'] == 'backward'`.

        There are few caveats to consider though. Invertible functions are hard
        to define without requiring more memory. Moreover, they are prone to
        numerical instabilities (e.g., multiplying by numbers close to zero).
        Even if we can get away with these fundamental problems, there are
        technical details to consider here. There is no way of guarding against
        accessing the data in the input tensors after calling the function and
        before the backward pass. It is up to the user to ensure this.
        Otherwise, it is possible to run into illegal memory access errors.
        Think of residual connections as an example. In `x + f(x)`, assuming
        `f` is an invertible checkpoint, `x` will be freed from memory before
        the sum is computed. On the other hand, we can maybe use
        `x.clone() + f(x)` (not `f(x) + x.clone()`!) but now we have a copy of
        `x` in memory. It is recommended to encapsulate this inside `f` itself
        or use the simple checkpoint instead. Other alternatives exists and you
        should study your case carefully before deciding to use this. For
        instance, check out `torch.autograd.graph.saved_tensors_hooks()` and
        `graph.save_on_cpu()`. Source: https://github.com/xmodar/invtorch

        Args:
            function: any forward function with `cache` argument
            inverse: inverse of `function` with `cache` argument
            keep: set of input tensors to keep in memory during inverse mode
            seed: same as preserve_rng_state; preserves random number generator
            enabled: disables checkpointing if set to False
            *args: input arguments `tuple` to be passed to `function`
            *kwargs: input keyword arguments `dict` to be passed to `function`

        Returns:
            Outputs of `function` with checkpointing if required
        """
        if not enabled or in_dry_mode() or not torch.is_grad_enabled():
            return function(*args, **kwargs)
        args = flat.args(args, kwargs)
        function, inverse = flat(function), flat(inverse)
        nonce = torch.tensor((), requires_grad=True)  # force differentiability
        return super().apply(function, inverse, pack(keep), seed, nonce, *args)

    @staticmethod
    def forward(ctx, function, inverse, keep, seed, _, *args):
        # capture autocast and RNG states and run the function in dry mode
        ctx.set_materialize_grads(False)
        ctx.autocast = torch.is_autocast_enabled()
        with dry_mode():
            ctx.forked_rng = DelayedRNGFork.from_tensors(args, seed)
            outputs = function({':mode': 'forward'}, False, *args)

        # bookkeep differentiable tensors
        grads = list(map(requires_grad, outputs))
        no_grads = (x for x, g in zip(outputs, grads) if not g)
        ctx.mark_non_differentiable(*filter(torch.is_tensor, no_grads))
        if not any(grads):
            return flat.outs(outputs)  # no output requires gradient

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
        return flat.outs(outputs)

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError('.grad()/.backward(inputs=...) not supported')

        # materialize any deallocated tensors by calling inverse
        tensors = ctx.saved_tensors
        if ctx.inverse is not None:
            saved = set(ctx.indices) - ctx.deallocated
            with torch.inference_mode():
                inverted = ctx.inverse({':mode': saved}, True, *ctx.outputs)
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
                outputs = ctx.function({':mode': 'backward'}, False, *inputs)
                outputs = pack(flat.outs(outputs))

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
