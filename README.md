# InvTorch: Memory-Efficient Invertible Functions

This module extends the functionality of `torch.utils.checkpoint.checkpoint` to work with invertible functions. So, not only the intermediate activations will be released from memory. The input tensors get deallocated and recomputed later using the inverse function only in the backward pass. This is useful in extreme situations where more compute is traded with memory. However, there are few caveats to consider which are detailed [here](./invtorch/utils/checkpoint.py).

## Installation

[InvTorch](https://github.com/xmodar/invtorch) has minimal dependencies. It only requires Python `>=3.6` and PyTorch `>=1.10.0`.

```bash
conda install pytorch==1.10.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install invtorch
```

## Basic Usage

The main module that we are interested in is `InvertibleModule` which inherits from `torch.nn.Module`. Subclass it to implement your own invertible code.

```python
import torch
from torch import nn

import invtorch.nn as inn
from invtorch.utils import requires_grad


class InvertibleLinear(inn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def function(self, inputs, strict=None):
        outputs = inputs @ self.weight.T + self.bias
        if strict:
            requires_grad(outputs, any=(inputs, self.weight, self.bias))
        return outputs

    def inverse(self, outputs, saved=()):
        if 0 in saved:
            return None
        return (outputs - self.bias) @ self.weight.T.pinverse()
```

### Structure

You can immediately notice few differences to the regular PyTorch module here. There is no longer a need to define `forward()`. Instead, it is replaced with `function(*inputs, strict=None)`. Additionally, it is necessary to define its inverse function as `inverse(*outputs, saved=())`. Both methods can only take one or more positional arguments and return a `torch.Tensor` or a `tuple` of outputs which can have anything including tensors.

### Requires Gradient

`function()` must manually call `.requires_grad_(True/False)` on all output tensors when `strict` is set to `True`. The forward pass is run in `no_grad` mode and there is no way to detect which output need gradients without tracing. It is possible to infer this from `requires_grad` values of the `inputs` and `self.parameters()`. The above code uses `invtorch.utils.require_grad(any=...)` which returns `True` if any input did require gradient. In `inverse()`, the keyword argument `saved` is passed. Which is the set of inputs positions that are already saved in memory and there is no need to compute them.

### Example

Now, this model is ready to be instantiated and used directly.

```python
x = torch.randn(10, 3)
model = InvertibleLinear(3, 5)
print('Consistent strict:', model.check_function(x))
print('Is invertible:', model.check_inverse(x))

y = model(x)
print('Output requires_grad:', y.requires_grad)
print('Input was freed:', x.storage().size() == 0)

y.backward(torch.randn_like(y))
print('Input was restored:', x.storage().size() != 0)
```

## Checkpoint and Invertible Modes

`InvertibleModule` has two flags which control the mode of operation; `checkpoint` and `invertible`. If `checkpoint` was set to `False`, or when working in `no_grad` mode, or no input or parameter has `requires_grad` set to `True`, it acts exactly as a normal PyTorch module. Otherwise, the model is either `invertible` or an ordinary `checkpoint` depending on whether `invertible` is set to `True` or `False`, respectively. Those, flags can be changed at any time during operation without any repercussions. A third flag `seed` is by default `False` and if set to `True`, it ensures that the forward runs in the same random number generator's state of the devices of the input tensors.

## Limitations

Under the hood, `InvertibleModule` uses `invtorch.checkpoint()`; a low-level implementation which allows it to function. It is an improved version of `torch.utils.checkpoint.checkpoint()`. There are few considerations to keep in mind when working with invertible checkpoints and non-materialized tensors. Please, refer to the [documentation](./invtorch/utils/checkpoint.py) in the code for more details.

## TODOs

Here are few feature ideas that could be implemented to enrich the utility of this package:

- Add more basic operations and modules
- Add coupling- and interleave-based invertible operations
- Add more checks to help the user debug more features
- Context-manager to temporarily change the mode of operation
- Implement dynamic discovery for outputs that requires_grad
- Develop an automatic mode optimization for a network for various objectives
