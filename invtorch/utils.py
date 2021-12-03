"""InvTorch: Utilities https://github.com/xmodar/invtorch"""
import functools

import torch

__all__ = ['requires_grad']

_any, _all = any, all


def requires_grad(inputs=None, *, any=None, all=None):
    """Check if the inputs is a tensor that requires_grad"""
    # pylint: disable=redefined-builtin
    if any is not None or all is not None:
        any = any is None or _any(map(requires_grad, any))
        all = all is None or _all(map(requires_grad, all))
        requires = any and all
        if inputs is not None:
            inputs.requires_grad_(requires)
        return requires
    else:
        assert inputs is not None, 'You have to pass inputs'
    return torch.is_tensor(inputs) and inputs.requires_grad


def pack(inputs, is_tensor=False):
    """Pack the inputs into tuple if they were a single tensor"""
    single = torch.is_tensor(inputs)
    outputs = (inputs, ) if single else inputs
    return (outputs, single) if is_tensor else outputs


def get_tensor_id(inputs, by_storage=True):
    """Get a uniquely identifying key for a tensor based on its storage"""
    assert torch.is_tensor(inputs)
    return inputs.storage().data_ptr() if by_storage else id(inputs)


def get_tensor_id_set(*inputs, by_storage=True):
    """Get a set of only the tensors' ids"""
    get_id = functools.partial(get_tensor_id, by_storage=by_storage)
    return set(map(get_id, filter(torch.is_tensor, inputs)))


class DelayedRNGFork:
    """Caputres PyTorch's RNG state on initialization with delayed use

    This context manager captures the torch random number generator states on
    instantiation. Then, it can be used many times later using with-statements
    Source: https://gist.github.com/xmodar/2328b13bdb11c6309ba449195a6b551a

    Example:
        rng = DelayedRNGFork(devices=[0])
        print('1 outside', torch.randn(3, device='cuda:0'))
        with rng:
            print('1 inside ', torch.randn(3, device='cuda:0'))
            print('2 inside ', torch.randn(3, device='cuda:0'))
        print('2 outside', torch.randn(3, device='cuda:0'))
    """
    def __init__(self, devices=None, enabled=True):
        self.cpu_state = torch.get_rng_state() if enabled else None
        self.gpu_states = {}
        if self.enabled:
            if devices is None:
                devices = range(torch.cuda.device_count())
            for device in map(torch.cuda.device, set(devices)):
                self.gpu_states[device.idx] = torch.cuda.get_rng_state(device)
        self._fork = None

    def __enter__(self):
        if self.enabled:
            self._fork = torch.random.fork_rng(self.gpu_states)
            self._fork.__enter__()  # pylint: disable=no-member
            torch.set_rng_state(self.cpu_state)
            for device, state in self.gpu_states.items():
                with torch.cuda.device(device):
                    torch.cuda.set_rng_state(state)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            fork, self._fork = self._fork, None
            # pylint: disable=no-member
            return fork.__exit__(exc_type, exc_value, traceback)
        return None

    @property
    def enabled(self):
        """Whether the fork is enabled"""
        return self.cpu_state is not None
