"""InvTorch: Improved Checkpoint https://github.com/xmodar/invtorch"""
import torch
from torch.utils.checkpoint import checkpoint as _checkpoint

from .utils import pack, requires_grad, get_tensor_id_set

__all__ = ['checkpoint']


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

    Instead, this function has two new flags; `strict` and `enabled`.
    Setting `enabled` to `False`, will disable all checkpointing logic. Else,
    when `strict` is set to `False`, it will use the original checkpoint as is.
    Running in `strict` mode incurs two main changes to the original behavior:
        (1) When no input tensor requires gradient but `function` generates
            outputs that do require gradients, the checkpoint will still work
        (2) All outputs will have `requires_grad` set to `False` by default.
            To specify what tensor actually requires gradient, `function` will
            expect a keyword argument `strict` which will be `True` only when
            we cannot automatically track this information. Here is an example:
            ```python
            def function(x, y, strict=False):
                z = x + y
                if strict:
                    z.requires_grad = x.requires_grad or y.requires_grad
                    # no need to set for y as it is already set
                return z, y
           ```
           Debug your code carefully and try to cover all the cases. Don't
           forget to account for the used parameters that requires_grad.

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

    # find out which mode to run? strict inverse, strict, or nonstrict
    strict = strict or inverse is not None  # inverse is always strict

    if not strict:  # use torch.utils.checkpoint.checkpoint
        return _checkpoint(function, *inputs, preserve_rng_state=seed)

    def _function(_, *inputs):
        strict = not torch.is_grad_enabled()
        outputs = function(*inputs, strict=strict)
        current = map(requires_grad, pack(outputs))
        if strict:  # get manually set requires_grad for output tensors
            grads.extend(current)
        else:  # and check them against automatically set requires_grad
            bad = {i for i, (g, c) in enumerate(zip(grads, current)) if g != c}
            if bad:
                expected = [i in bad != g for i, g in enumerate(grads)]
                msg = ('manually set requires_grad for output tensors in '
                       'strict mode mismatched automatically set values in '
                       'the backward pass. Please, debug your implementation '
                       'carfully and try to cover all the cases. Keep in mind '
                       'the paramters of any model you call in `function`.'
                       f'\nExpected: {expected}\nReceived: {grads}')
                raise RuntimeError(msg)
        return outputs

    grads = []  # to be filled by `_function` with requires_grad of outputs
    nonce = torch.tensor((), requires_grad=True)  # ensures differentiability
    outputs = _checkpoint(_function, nonce, *inputs, preserve_rng_state=seed)

    for grad, argument in zip(grads, pack(outputs)):
        if torch.is_tensor(argument) and not grad:
            argument.detach_()  # detach tensors that don't require gradients
    if not any(grads):  # apparently, `function` was not differentiable
        return outputs

    if inverse is not None:  # see if we really need inverse
        keep = get_tensor_id_set(*keep, *pack(outputs))
        seep = get_tensor_id_set(*inputs) - keep  # tensors to marshal
        if not seep:  # if we need to keep all input tensors
            inverse = None  # then, ignore the inverse function
    if inverse is None:
        return outputs

    raise NotImplementedError('`inverse` is not currently supported')
