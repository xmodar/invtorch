"""Gradient Modes and InvTorch State"""
import threading
from contextlib import contextmanager

import torch

__all__ = ['dry_mode', 'in_dry_mode']

_local = threading.local()
_local.dry_mode = 0


def in_dry_mode():
    """Whether we are in dry_mode"""
    return _local.dry_mode != 0


def _dry_mode():
    def generator():
        yield

    if not in_dry_mode():
        unpack = lambda _: generator().throw(
            RuntimeError('cannot run backward on tensors produced in dry mode')
        )
        return torch.autograd.graph.saved_tensors_hooks(lambda _: None, unpack)
    return contextmanager(generator)()  # no-op context manager


@contextmanager
def dry_mode(enabled=True):
    """Enable dry_mode; gradients are enabled but backward is not allowed"""
    if enabled:
        try:
            with _dry_mode(), torch.enable_grad():
                _local.dry_mode += 1
                yield
        finally:
            _local.dry_mode -= 1
    else:
        yield
