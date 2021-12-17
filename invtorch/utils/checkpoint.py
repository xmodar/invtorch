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
