# InvTorch: Memory-Efficient Invertible Functions

This module extends the functionality of `torch.utils.checkpoint.checkpoint` to work with invertible functions. So, not only the intermediate activations will be released from memory. The input tensors get deallocated and recomputed later using the inverse function only in the backward pass. This is useful in extreme situations where more compute is traded with memory. However, there are few considerations to keep in mind when working with invertible checkpoints and non-materialized tensors. Please, refer to the [documentation](./invtorch/utils/checkpoint.py) in the code for more details.

## Installation

[InvTorch](https://github.com/xmodar/invtorch) has minimal dependencies. It only requires Python `>=3.6` and PyTorch `>=1.10.0`.

```bash
conda install pytorch==1.10.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install invtorch
```

## Basic Usage

We are interested in `invtorch.nn.Module` which inherits from `torch.nn.Module`. Subclass it to implement your own invertible code. Refer to [this](./invtorch/nn/modules) for better examples.

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
        return (outputs - self.bias) @ self.weight.T.pinverse()
```

### forward()

You can immediately notice few differences to the regular PyTorch module here. There is no longer a need to define `forward()`. Instead, it is replaced with `function(*inputs, strict=None)`. Additionally, it is necessary to define its inverse function as `inverse(*outputs, saved=())`. Both methods can only take one or more positional arguments and return a `torch.Tensor` or a `tuple` of outputs which can have anything including tensors.

### function()

The first call to `function()` is always run in `no_grad` mode. So, there is no cheap way of detecting which output needs gradients. It is possible to infer this from `requires_grad` values of the `inputs` and the parameters. Therefore, `function()` must manually call `.requires_grad_(True/False)` on all output tensors when `strict` is set to `True`. You can use `invtorch.utils.require_grad(any=...)` which returns `True` if any input did require gradient. You can verify your implementation by simply calling `check_function()`.

### inverse()

In `inverse()`, the keyword argument `saved` is passed. It is a set of inputs positions of the tensors that are already saved in memory and there is no need to recompute them. It can be completely ignored if the number of inputs to `function()` is one since `inverse()` will not be called unless needed. You can verify your implementation by calling `check_inverse()`.

### reverse()

`invtorch.nn.Module` can be implemented to be reversible, i.e. `forward()` will call `inverse()` instead of `function()`. Not all invertible modules need to support reversibility. If you want to support it in your own module, then you need to override the `reversible` property to return `True`. Also, let both `function()` and `inverse()` accept each other's arguments; `strict` and `saved`. The module can be revered by calling `reverse()` and checked with the `reversed` property. To avoid confusion, `Module` has `call_function()` and `call_inverse()` which will call the correct function based on the `reversed` value.

### process_outputs()

Sometimes, `inverse()` needs some outputs that are not necessarily needed as an output of `forward()`. For example, batch normalization will need `mean` and `var` as output to be fed to `inverse()`. `forward()` will call `process_outputs()` in the background to get rid of this extra outputs. It will know what to keep by the `num_outputs` attribute which is inferred from `num_function_outputs` and `num_inverse_outputs` attributes depending on the `reversed` value. If `num_outputs` was `None`, all outputs will be used. On the other hand, if it was negative, its absolute value represent the number of extra variables.

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

`invtorch.nn.Module` has two flags which control the mode of operation; `checkpoint` and `invertible`. If `checkpoint` was set to `False`, or when working in `no_grad` mode, it acts exactly as a normal PyTorch module. Otherwise, the model is either `invertible` or an ordinary `checkpoint` depending on whether `invertible` is set to `True` or `False`, respectively. Those, flags can be changed at any time during operation without any repercussions. A third flag `seed` is by default `False` and if set to `True`, it ensures that the forward runs in the same random number generator's state of the devices of the input tensors.

## TODOs

Here are few feature ideas that could be implemented to enrich the utility of this package:

- Support older versions of PyTorch
- Add more basic operations and modules
- Add coupling- and interleave-based invertibles
- Add more checks to help the user debug more features
- Context-manager to temporarily change the mode of operation
- Implement dynamic discovery for outputs that requires_grad
- Develop an automatic [mode optimization](https://arxiv.org/abs/1604.06174) for a network
