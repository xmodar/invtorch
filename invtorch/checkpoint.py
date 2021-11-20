"""InvTorch: Improved Checkpoint https://github.com/xmodar/invtorch"""
import torch
from torch.utils.checkpoint import checkpoint as _checkpoint

from .utils import (DelayedRNGFork, get_tensor_id, get_tensor_id_set, pack,
                    requires_grad)

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

    def _function(_, *inputs):
        strict = not torch.is_grad_enabled()
        outputs = function(*materialize(inputs), strict_forward=strict)
        assert isinstance(outputs, (torch.Tensor, tuple)), 'wrong output type'
        current = map(requires_grad, pack(outputs))
        if strict:  # get manually set requires_grad for output tensors
            grads.extend(current)
        else:  # and check them against automatically set requires_grad
            bad = {i for i, (g, c) in enumerate(zip(grads, current)) if g != c}
            if bad:
                expected = [(i in bad) != g for i, g in enumerate(grads)]
                msg = ('manually set requires_grad for output tensors in '
                       'strict mode mismatched automatically set values in '
                       'the backward pass. Please, debug your implementation '
                       'carfully and try to cover all the cases. Keep in mind '
                       'the paramters of any model you call in `function`.'
                       f'\nExpected: {expected}\nReceived: {grads}')
                raise RuntimeError(msg)
        return outputs

    grads = []  # to be filled by `_function` with requires_grad of outputs
    materialize = lambda x: x  # dummy stub replaced and used in inverse mode
    nonce = torch.tensor((), requires_grad=True)  # ensures differentiability
    outputs = _checkpoint(_function, nonce, *inputs, preserve_rng_state=seed)

    # detach tensors that don't require gradients
    packed_outputs = []
    for grad, argument in zip(grads, pack(outputs)):
        if torch.is_tensor(argument) and not grad:
            argument = argument.detach()
        packed_outputs.append(argument)
    packed_outputs = tuple(packed_outputs)
    outputs = packed_outputs[0] if torch.is_tensor(outputs) else packed_outputs

    if not any(grads):  # apparently, `function` was not differentiable
        return outputs

    if inverse is not None:  # see if we really need inverse
        keep = get_tensor_id_set(*keep, *packed_outputs)
        seep = get_tensor_id_set(*inputs) - keep  # tensors to marshal
        if not seep:  # if we need to keep all input tensors
            inverse = None  # then, ignore the inverse function
    if inverse is None:
        return outputs

    def materialize(inputs):  # pylint: disable=function-redefined
        with torch.no_grad():
            inverted = pack(inverse(*packed_outputs))
            assert isinstance(inverted, tuple), 'inverse has wrong output type'
        inputs = _Materialize.apply(deallocated, inverted, *inputs)
        if seed:
            torch.set_rng_state(cpu_rng_state)
            for device, gpu_rng_state in zip(rng_devices, gpu_rng_states):
                torch.cuda.set_rng_state(gpu_rng_state, device)
        return inputs

    if seed:
        ctx = packed_outputs[grads.index(True)].grad_fn
        cpu_rng_state = ctx.fwd_cpu_state
        rng_devices = ctx.fwd_gpu_devices if ctx.had_cuda_in_fwd else []
        gpu_rng_states = ctx.fwd_gpu_states if ctx.had_cuda_in_fwd else []

    deallocated = set()
    for i, tensor in enumerate(inputs):
        if torch.is_tensor(tensor) and get_tensor_id(tensor) in seep:
            tensor.storage().resize_(0)  # deallocate the tensor
            deallocated.add(i)
    return outputs


class _Materialize(torch.autograd.Function):
    """Don't use! This function is intended only for checkpoint()"""
    # pylint: disable=abstract-method, arguments-differ
    @staticmethod
    def forward(ctx, deallocated, inverted, *inputs):
        ctx.set_materialize_grads(False)
        outputs, non_differentiable = [], []
        for i, (original, tensor) in enumerate(zip(inputs, inverted)):
            outputs.append(tensor if i in deallocated else original)
            if not original.requires_grad:
                non_differentiable.append(tensor)
        ctx.mark_non_differentiable(*non_differentiable)
        return tuple(outputs)

    @staticmethod
    def backward(_, *grads):
        return (None, None) + grads


class CheckpointFunction(torch.autograd.Function):
    """Same as `torch.utils.checkpoint.CheckpointFunction`"""
    # pylint: disable=abstract-method, arguments-differ, protected-access
    @staticmethod
    def forward(ctx, function, preserve_rng_state, *args):
        devices = (x.device for x in args if torch.is_tensor(x) and x.is_cuda)
        ctx.function = function
        ctx.autocast = torch.is_autocast_enabled()
        ctx.rng = DelayedRNGFork(devices, preserve_rng_state)
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)
        ctx.save_for_backward(*tensor_inputs)
        with torch.no_grad():
            return function(*args)

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError('.grad()/.backward(inputs=...) not supported')
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i].detach()
            inputs[idx].requires_grad_(tensors[i].requires_grad)
        with ctx.rng:
            with torch.enable_grad(), torch.cuda.amp.autocast(ctx.autocast):
                outputs = pack(ctx.function(*inputs))
        outputs_with_grad = []
        args_with_grad = []
        for i, output in enumerate(outputs):
            if requires_grad(output):
                outputs_with_grad.append(output)
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError('no output has requires_grad=True')
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = (x.grad if torch.is_tensor(x) else None for x in inputs)
        return (None, None, *grads)
