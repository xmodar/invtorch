"""Random Number Generator Utilities"""
import torch

__all__ = ['DelayedRNGFork']


class DelayedRNGFork:
    """Caputres PyTorch's RNG state on initialization with delayed use

    This context manager captures the torch random number generator states on
    instantiation. Then, it can be used many times later using with-statements

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
                torch.cuda.set_rng_state(state, device)

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

    @classmethod
    def from_tensors(cls, *args, enabled=True):
        """Get delayed RNG fork from input tensors"""
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        devices = (x.device for x in args if torch.is_tensor(x) and x.is_cuda)
        return cls(devices, enabled)
