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


class InvertibleLinear(inn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def function(self, inputs, cache=None):
        return inputs @ self.weight.T + self.bias

    def inverse(self, outputs, cache=None):
        return (outputs - self.bias) @ self.weight.T.pinverse()
```

### forward()

You can immediately notice few differences to the regular PyTorch module here. There is no longer a need to define `forward()`. Instead, it is replaced with `function()`. Additionally, it is necessary to define its inverse function as `inverse()`. Both methods can take any number of arguments including a keyword argument `cache`. This will be a `dict` to be filled with the keyword arguments of `inverse()`. Both function should return a `torch.Tensor` or a `tuple` of outputs which can have anything including tensors.

### function()

The first call to `function()` is always run in `dry_mode`. This is a novel mode that has gradient graph construction enabled but doesn't allow for backward propagation. The second call is during the backward pass which is when the gradients will actually be computed. The argument `cache[':mode']` will be `'forward'` in the first call and `'backward'` in the second call. Your function should handle the case when `None` is passed instead.

### inverse()

You can verify your implementation of `inverse()` by calling `check_inverse()`. In some cases, switching to double precision is advised as invertible functions can run into some numerical instability when using single precision.

### reverse()

`invtorch.nn.Module` can be implemented to be reversible, i.e. `forward()` will call `inverse()` instead of `function()`. Not all invertible modules need to support reversibility. If you want to support it in your own module, then you need to override the `reversible` property to return `True`. Also, let both `function()` and `inverse()` accept each other's arguments; `strict` and `saved`. The module can be revered by calling `reverse()` and checked with the `reversed` property. To avoid confusion, `Module` has `call_function()` and `call_inverse()` which will call the correct function based on the `reversed` value.

### Example

Now, this model is ready to be instantiated and used directly.

```python
x = torch.randn(10, 3)
model = InvertibleLinear(3, 5)
print('Is invertible:', model.check_inverse(x))

y = model(x)
print('Input was freed:', x.storage().size() == 0)

y.backward(torch.randn_like(y))
print('Input was restored:', x.storage().size() != 0)
```

## Checkpoint and Invertible Modes

`invtorch.nn.Module` has two flags which control the mode of operation; `checkpoint` and `invertible`. If `checkpoint` was set to `False`, or when working in `no_grad` mode, it acts exactly as a normal PyTorch module. Otherwise, the model is either `invertible` or a `checkpoint` depending on whether `invertible` is set to `True` or `False`, respectively. Those, flags can be changed at any time during operation without any repercussions. A third flag `seed` is by default `False` and if set to `True`, it ensures that the forward runs in the same random number generator's state of the devices of the input tensors.

## TODOs

Here are few feature ideas that could be implemented to enrich the utility of this package:

- Add more basic operations and modules
- Add coupling- and interleave-based invertibles
- Add more checks to help the user debug more features
- Develop an automatic [mode optimization](https://arxiv.org/abs/1604.06174) for a network
