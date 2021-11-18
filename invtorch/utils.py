"""InvTorch: Utilities https://github.com/xmodar/invtorch"""
import functools
import torch


def pack(inputs, is_tensor=False):
    """Pack the inputs into tuple if they were a single tensor"""
    single = torch.is_tensor(inputs)
    outputs = (inputs, ) if single else inputs
    return (outputs, single) if is_tensor else outputs


def requires_grad(inputs):
    """Check if the inputs is a tensor that requires_grad"""
    return torch.is_tensor(inputs) and inputs.requires_grad


def get_tensor_id(inputs, by_storage=True):
    """Get a uniquely identifying key for a tensor based on its storage"""
    assert torch.is_tensor(inputs)
    return inputs.storage().data_ptr() if by_storage else id(inputs)


def get_tensor_id_set(*inputs, by_storage=True):
    """Get a set of only the tensors' ids"""
    get_id = functools.partial(get_tensor_id, by_storage=by_storage)
    return set(map(get_id, filter(torch.is_tensor, inputs)))
