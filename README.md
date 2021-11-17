# InvTorch: Memory-Efficient Invertible Functions

This module extends the functionality of `torch.utils.checkpoint.checkpoint` to work with invertible functions. So, not only the intermediate activations will be released from memory. The input tensors get deallocated and recomputed later using the inverse function only in the backward pass. This is useful in extreme situations where more compute is traded with memory. However, there are few caveats to consider which are detailed [here](./invtorch/core.py).

## Installation

InvTorch has minimal dependencies. It only requires PyTorch version `1.10.0` or later.

```bash
conda install pytorch==1.10.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install git+https://github.com/xmodar/invtorch
```

## Basic Usage

The main module that we are interested in is `InvertibleModule` which inherits from `torch.nn.Module`. Subclass it to implement your own invertible code.

```python
import torch
from torch import nn
from invtorch import InvertibleModule


class InvertibleLinear(InvertibleModule):
    def __init__(self, in_features, out_features):
        super().__init__(invertible=True, checkpoint=True)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def function(self, inputs):
        outputs = inputs @ self.weight.T + self.bias
        requires_grad = self.do_require_grad(inputs, self.weight, self.bias)
        return outputs.requires_grad_(requires_grad)

    def inverse(self, outputs):
        return (outputs - self.bias) @ self.weight.T.pinverse()
```

### Structure

You can immediately notice few differences to the regular PyTorch module here. There is no longer a need to define `forward()`. Instead, it is replaced with `function(*inputs)`. Additionally, it is necessary to define its inverse function as `inverse(*outputs)`. Both methods can only take one or more positional arguments and return a `torch.Tensor` or a `tuple` of outputs which can have anything including tensors.

### Requires Gradient

`function()` must manually call `.requires_grad_(True/False)` on all output tensors. The forward pass is run in `no_grad` mode and there is no way to detect which output need gradients without tracing. It is possible to infer this from `requires_grad` values of the `inputs` and `self.parameters()`. The above code uses `do_require_grad()` which returns `True` if any input did require gradient.

### Example

Now, this model is ready to be instantiated and used directly.

```python
x = torch.randn(10, 3)
model = InvertibleLinear(3, 5)
print('Is invertible:', model.check_inverse(x))

y = model(x)
print('Output requires_grad:', y.requires_grad)
print('Input was freed:', x.storage().size() == 0)

y.backward(torch.randn_like(y))
print('Input was restored:', x.storage().size() != 0)
```

## Checkpoint and Invertible Modes

`InvertibleModule` has two flags which control the mode of operation; `checkpoint` and `invertible`. If `checkpoint` was set to `False`, or when working in `no_grad` mode, or no input or parameter has `requires_grad` set to `True`, it acts exactly as a normal PyTorch module. Otherwise, the model is either `invertible` or an ordinary `checkpoint` depending on whether `invertible` is set to `True` or `False`, respectively. Those, flags can be changed at any time during operation without any repercussions.

## Limitations

Under the hood, `InvertibleModule` uses `invertible_checkpoint()`; a low-level implementation which allows it to function. There are few considerations to keep in mind when working with invertible checkpoints and non-materialized tensors. Please, refer to the [documentation](./invtorch/core.py) in the code for more details.

## Overriding `forward()`

Although `forward()` is now doing important things to ensure the validity of the results when calling `invertible_checkpoint()`, it can still be overridden. The main reason of doing so is to provide a more user-friendly interface; function signature and output format. For example, `function()` could return extra outputs that are not needed in the module outputs but are essential for correctly computing the `inverse()`. In such case, define `forward()` to wrap `outputs = super().forward(*inputs)` more cleanly.

## TODOs

Here are few feature ideas that could be implemented to enrich the utility of this package:

- Add more basic operations and modules
- Add coupling and interleave -based invertible operations
- Add more checks to help the user in debugging more features
- Allow picking some inputs to not be freed in invertible mode
- Context-manager to temporarily change the mode of operation
- Implement dynamic discovery for outputs that requires_grad
- Develop an automatic mode optimization for a network for various objectives
