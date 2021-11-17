"""InvTorch: Core Invertible Utilities https://github.com/xmodar/invtorch"""
import itertools
import collections

import torch
import torch.utils.checkpoint

__all__ = ['invertible_checkpoint', 'InvertibleModule']


def invertible_checkpoint(function, inverse, *args, preserve_rng_state=False):
    """Checkpoint a model or part of the model without saving the input

    Extends the functionality of `torch.utils.checkpoint.checkpoint` to work
    with invertible functions. So, not only the intermediate activations can be
    released from memory but also the input tensors as well. They will get
    recomputed using the inverse function in the backward pass. This is useful
    in extreme situations where more compute is traded with memory. However,
    there are few caveats to consider here. Invertible functions are hard to
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
    use a simple checkpoint instead. For similar reasons, identity functions
    and functions with output views of the inputs should not be used here.

    This function was implemented with PyTorch 1.10.0 in mind. It has the same
    inputs as the simple checkpoint except for the `inverse` function and the
    default value for `preserve_rng_state` is False. Other than that the same
    restrictions that apply to checkpoint apply here as well. For example, it
    doesn't support `torch.autograd.grad()` and `backward(inputs=...)`. Also,
    at least one input and output tensor should have requires_grad set to True.
    In addition, all outputs will have their requires_grad set to True even if
    they normally wouldn't and those should be detached manually. Furthermore,
    it goes without saying that `function` shouldn't modify the input in-place.
    Refer to https://pytorch.org/docs/1.10.0/checkpoint.html for more details.

    There are other considerations concerning the `inverse` function. It will
    be run in `torch.inference_mode()`. It shouldn't have side effects and must
    not modify the outputs (its inputs) in-place in most cases. It only needs
    to compute the input tensors and can return anything for non-tensor inputs
    but it is a good practice to return them to allow for nesting invertible
    functions. Finally, the inputs and outputs to `function` and `inverse` must
    either be single tensors or a tuple (not a list) to guarantee correctness.
    Source: https://github.com/xmodar/invtorch

    Args:
        function: invertible differentiable function
        inverse: inverse of `function` (doesn't need to be differentiable)
        *args: input arguments tuple to be passed to `function`
        preserve_rng_state: use same seed when calling `function` in backward

    Returns:
        Outputs of `function(*args)` with requires_grad=True for all tensors
    """
    def unpack(index):
        counts[index] -= 1
        if unpack.outputs is not None:
            with torch.inference_mode():
                inverted = unpack.inverse(*unpack.outputs)
                unpack.outputs = unpack.inverse = None
            inverted = (inverted, ) if torch.is_tensor(inverted) else inverted
            assert len(inverted) == len(args), 'inverse(outputs) != inputs'
            list(tensors[i].set_(x) for i, x in zip(args, inverted))
        return tensors[index] if counts[index] else tensors.pop(index)

    # the following code might seem strange but it was intentionally written
    # this way to avoid dangling references in the unpack function to any
    # object that isn't crucially needed for materializing the input correctly
    # I know that it can be refactored much more elegantly if we allow unpack
    # to be defined outside, but it is self-contained and short enough ;)
    preserve_rng_state = bool(preserve_rng_state)
    tensors = {id(x): x for x in args if torch.is_tensor(x)}
    assert any(x.requires_grad for x in tensors.values()), 'no input need grad'
    with torch.autograd.graph.saved_tensors_hooks(id, unpack):
        unpack.outputs = torch.utils.checkpoint.checkpoint(
            function, *args, preserve_rng_state=preserve_rng_state)
        function, inverse, unpack.inverse = None, None, inverse
        args = [id(x) if torch.is_tensor(x) else None for x in args]
        counts = dict(collections.Counter(args))
    single = torch.is_tensor(unpack.outputs)
    unpack.outputs = (unpack.outputs, ) if single else unpack.outputs
    assert isinstance(unpack.outputs, tuple), 'function must return a tuple'
    list(x.storage().resize_(0) for x in tensors.values())
    return unpack.outputs[0] if single else unpack.outputs


class InvertibleModule(torch.nn.Module):
    """Base invertible `inputs = self.inverse(*self.function(*inputs))`

    Use this with great caution. Refer to the notes in invertible_checkpoint()
    Source: https://github.com/xmodar/invtorch
    """
    def __init__(self, invertible=True, checkpoint=True):
        super().__init__()
        self.invertible = invertible  # use inverse if checkpointing is enabled
        self.checkpoint = checkpoint  # enables or disables checkpointing

    def function(self, *inputs):
        """Compute the outputs of the function given the inputs

        The first run of function will be in no_grad mode. Thus, you must
        manually call `.requires_grad_(True/False)` for all output tensors.
        It is possible to infer the values from requires_grad of `inputs` and
        `self.parameters()`. You should handle all possible combinations or
        you will get unexpected behavior. You can verify your implementation
        by simply calling this function with different inputs and checking
        requires_grad of the output tensors before setting them yourself.
        """
        raise NotImplementedError

    def inverse(self, *outputs):
        """Compute the inputs of the function given the outputs

        Verify your implementation by calling `self.check_inverse()`
        """
        raise NotImplementedError

    @torch.inference_mode()
    def check_inverse(self, *inputs, atol=1e-5, rtol=1e-3):
        """Check if `self.inverse()` is correct for input tensors"""
        outputs = self.pack(self.inverse(*self.pack(self.function(*inputs))))
        for inputs, outputs in itertools.zip_longest(inputs, outputs):
            is_tensor = torch.is_tensor(inputs)
            assert is_tensor == torch.is_tensor(outputs)
            assert not is_tensor or torch.allclose(inputs, outputs, rtol, atol)
        return True

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

    def forward(self, *inputs):
        """Perform the forward pass"""
        if not self.checkpoint or not torch.is_grad_enabled() or not any(
                True for x in itertools.chain(self.parameters(), inputs)
                if torch.is_tensor(x) and x.requires_grad):
            return self.function(*inputs)
        if self.invertible:
            apply = invertible_checkpoint
        else:
            apply = torch.utils.checkpoint.checkpoint
        zero = torch.zeros((), requires_grad=True)  # ensure differentiability
        grads, one, *out = apply(self._function, self._inverse, zero, *inputs)
        for outputs, requires_grad in zip(out, grads):
            if torch.is_tensor(outputs) and not requires_grad:
                outputs.detach_()
        return out[0] if one.item() else out

    def _function(self, _, *inputs):
        """Wraps `self.function` to handle no requires_grad inputs"""
        outputs, one = self.pack(self.function(*inputs), True)
        grads = [torch.is_tensor(x) and x.requires_grad for x in outputs]
        one = torch.tensor(float(one), requires_grad=True)
        return (grads, one, *outputs)

    def _inverse(self, _, one, *outputs):
        """Wraps `self.inverse` to handle no requires_grad inputs"""
        return (one, *self.pack(self.inverse(*outputs)))

    @staticmethod
    def pack(inputs, is_tensor=False):
        """Pack the inputs into tuple if they were a one tensor"""
        one = torch.is_tensor(inputs)
        outputs = (inputs, ) if one else inputs
        return (outputs, one) if is_tensor else outputs

    @staticmethod
    def do_require_grad(*tensors, at_least_one=True):
        """Check whether input tensors have `requires_grad=True`"""
        for tensor in tensors:
            requires_grad = torch.is_tensor(tensor) and tensor.requires_grad
            if at_least_one == requires_grad:
                return at_least_one
        return not at_least_one
