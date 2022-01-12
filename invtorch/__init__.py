"""InvTorch: Memory-Efficient Invertible Functions

Sources:
    https://github.com/xmodar/invtorch
    https://github.com/silvandeleemput/memcnn
    https://gist.github.com/xmodar/4deb8905ed8c294862972466f69e5d17
    https://gist.github.com/xmodar/2328b13bdb11c6309ba449195a6b551a
    https://gist.github.com/xmodar/7921460648230eda5053fe06b7cd2f4d
"""
from . import autograd, nn, random, utils
from .autograd.grad_mode import *
from .utils.checkpoint import *

__version__ = '0.5.0'
