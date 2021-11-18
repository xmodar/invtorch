"""InvTorch: Improved Checkpoint"""
import torch
from torch.utils.checkpoint import checkpoint as _checkpoint

__all__ = ['checkpoint']


def checkpoint(function, *inputs, seed=True, strict=False, enabled=True):
    """Same as `torch.utils.checkpoint.checkpoint` with extra functionalities

    The original checkpoint needs at least one input with `requires_grad` set
    to `True` to work. This is fine as long as the `function` doesn't have any
    tensors that require gradients. In such case, no gradients will be computed
    and PyTorch will raise a UserWarning. The checkpoint will be disabled and
    the code will run in `torch.no_grad()` mode.

    In addition, by default all output tensors will have `requires_grad` set
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
                z = x * y
                if strict:
                    z.requires_grad = x.requires_grad or y.requires_grad
                return z
           ```
           Debug your code carefully and try to cover all the cases. Don't
           forget to account for the used parameters that requires_grad.

    Args:
        function: this is any forward function with positional arguments
        *args: input arguments tuple to be passed to `function`
        seed: same as preserve_rng_state; preserves the random number generator
        strict: `requires_grad` for outputs is set manually in `function`
        enabled: disables checkpointing if set to False

    Returns:
        Outputs of `function(*args)` with requires_grad=True for all tensors
    """
    if not enabled or not torch.is_grad_enabled():  # no checkpointing
        return function(*inputs)
    kwargs = dict(preserve_rng_state=seed)
    if not strict:  # use torch.utils.checkpoint.checkpoint
        return _checkpoint(function, *inputs, **kwargs)

    def _function(_, *args):
        outputs = function(*args, strict=not torch.is_grad_enabled())
        single = torch.is_tensor(outputs)  # return a single tensor or a tuple
        if single:
            outputs = (outputs, )
        # get which output arguments were set manually to require gradients
        grads = [torch.is_tensor(x) and x.requires_grad for x in outputs]
        return (single, grads, *outputs)

    nonce = torch.tensor((), requires_grad=True)  # ensures differentiability
    single, grads, *outputs = _checkpoint(_function, nonce, *inputs, **kwargs)

    for requires_grad, argument in zip(grads, outputs):
        if torch.is_tensor(argument) and not requires_grad:
            argument.detach_()

    return outputs[0] if single else outputs
