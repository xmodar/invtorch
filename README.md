# InvTorch: Memory-Efficient Invertible Functions

When working with extremely deep neural networks, memory becomes an immediate concern. It will be filled with [saved tensors](https://pytorch.org/docs/1.10.0/notes/autograd.html#:~:text=Saved%20tensors) that are needed for gradient computation. PyTorch provides a solution, [`checkpoint_sequential`](https://pytorch.org/docs/1.10.0/checkpoint.html#:~:text=torch.utils.checkpoint.checkpoint_sequential), that allows us to segment every few layers as a checkpoint. Such that, the forward pass is run in `no_grad` mode. Meanwhile, the inputs of every segment is saved in memory. Every other unreferenced tensor gets deallocated. In the backward pass, the forward pass of every segment, starting from the last to the first, will be run again using its saved inputs to compute its gradients. Refer to [this](https://pytorch.org/docs/1.10.0/checkpoint.html) for more details.

This module extends the functionality of `torch.utils.checkpoint.checkpoint` to work with invertible functions. So, everything now can be released from memory and recomputed later using the inverse function in the backward pass. This is useful for extremely wide networks where more compute is traded with memory. However, there are few considerations to keep in mind when working with invertible checkpoints and non-materialized tensors. Please, read the limitations section below for specifics.

## Installation

[InvTorch](https://github.com/xmodar/invtorch) has minimal dependencies. It only requires Python `>=3.6` and PyTorch `>=1.10.0`.

```bash
pip install invtorch
```

## Usage

We are interested in `invtorch.nn.Module` which inherits from `torch.nn.Module`. Subclass it to implement your own invertible code.

```python
import torch
from torch import nn

import invtorch.nn as inn


class InvertibleLinear(inn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def function(self, inputs):
        return inputs @ self.weight.T + self.bias

    def inverse(self, outputs):
        return (outputs - self.bias) @ self.weight.T.pinverse()


if __name__ == '__main__':
    x = torch.randn(10, 3)
    model = InvertibleLinear(3, 5)
    print('Is invertible:', model.check(x))

    y = model(x)
    print('Input was freed:', x.storage().size() == 0)

    y.backward(torch.randn_like(y))
    print('Input was restored:', x.storage().size() != 0)
```

### forward()

You can immediately notice few differences to the regular PyTorch module here. There is no longer a need to define `forward()`. Instead, it is replaced with `function()`. Additionally, it is necessary to define its inverse function as `inverse()`. Both methods should input and output only positional arguments as a `tuple` or a single `torch.Tensor`. Furthermore, `forward()` accepts an optional keyword argument `keep` which is an iterable of input tensors that shouldn't be deallocated.

### function()

The first call to `function()` is always run in `dry_mode` (check it with `invtorch.in_dry_mode()`). This is a novel mode that has gradient graph construction enabled but doesn't allow for backward propagation. The second call is during the backward pass which is when the gradients will actually be computed (check it with `invtorch.in_backward()`).

### inverse()

You can verify your implementation of `inverse()` by calling `check()`. In some cases, switching to double precision is advised as invertible functions can run into some numerical instability when using single precision. For some functions, a view of an input tensor is passed in the outputs. In such case, this will be automatically detected and the input tensor will not be released from memory.

### num_outputs

Sometimes, `inverse()` requires auxiliary outputs from `function()` that are not necessarily needed as an output for `forward()`. To specify how many outputs do we need to keep, we can modify the `num_outputs` property (`None` keeps everything). Extra outputs will be hidden using `process()` which is automatically called by `forward()`. Another property `num_inputs` is also defined which helps composing inverse calls like in `invtorch.nn.Sequential.inverse()`.

### reverse()

`invtorch.nn.Module` can be implemented to be reversible, i.e. `forward()` will call `inverse()` instead of `function()`. Not all invertible modules need to support reversibility. If you want to support it on your own module, then you need to override the `reversible` property to return `True`. The module can be reversed by calling `reverse()` and checked with the `reversed` property. To avoid confusion, `Module` has `call_function()` and `call_inverse()` which will call the correct function based on the value of `reversed`.

### checkpoint, invertible, and seed

`invtorch.nn.Module` has two flags which control the mode of operation; `checkpoint` and `invertible`. If `checkpoint` was set to `False`, or when working in `no_grad` mode, it acts exactly as a normal PyTorch module. Otherwise, the model is either `invertible` or a `checkpoint` depending on whether `invertible` is set to `True` or `False`, respectively. Those, flags can be changed at any time during operation without any repercussions. A third flag `seed` is by default `False` and if set to `True`, it ensures that the forward runs in the same random number generator's state of the devices of the input tensors.

### invtorch.checkpoint()

PyTorch's checkpoint cannot track the `requires_grad` attribute for its output tensors since it is running in `no_grad` mode. Instead, InvTorch's checkpoint doesn't have this issue because it runs in `dry_mode`. In addition, it supports invertible functions.

## Limitations

There are few caveats to consider though. Invertible functions are hard to define without requiring more memory. Moreover, they are prone to numerical instabilities (e.g., multiplying by numbers close to zero). Even if we can get away with these fundamental problems, there are technical details to consider. There is no way of guarding against accessing the data in the input tensors after calling `function()` and before the backward pass. It is up to the user to ensure this. Otherwise, it is possible to run into illegal memory access errors. Think of residual connections as an example. In `x + f(x)`, assuming `f` is an invertible checkpoint, `x` will be freed from memory before the sum is computed. On the other hand, we can maybe use `x.clone() + f(x)` (not `f(x) + x.clone()`!) but now we have a copy of `x` in memory. It is recommended to encapsulate this inside `f` itself or use a simple checkpoint instead. Other alternatives exists and you should study your case carefully before deciding to use this. For instance, check out `torch.autograd.graph.saved_tensors_hooks()` and `torch.autograd.graph.save_on_cpu()`.
