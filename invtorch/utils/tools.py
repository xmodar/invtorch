"""Invertible Tools"""
import builtins

import torch

__all__ = ['pack', 'requires_grad']


def pack(inputs, is_tensor=False):
    """Pack the inputs into tuple if they were a single tensor"""
    single = torch.is_tensor(inputs)
    outputs = (inputs, ) if single else inputs
    return (outputs, single) if is_tensor else outputs


def requires_grad(inputs=None, *, any=None, all=None):
    """Check if the inputs is a tensor that requires_grad"""
    # pylint: disable=redefined-builtin
    if any is not None or all is not None:
        any = any is None or builtins.any(map(requires_grad, any))
        all = all is None or builtins.all(map(requires_grad, all))
        requires = any and all
        if inputs is not None:
            inputs.requires_grad_(requires)
        return requires
    return torch.is_tensor(inputs) and inputs.requires_grad
