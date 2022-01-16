"""Invertible PixelShuffle Modules"""
import itertools

from torch import nn

from .module import Module

__all__ = ['PixelShuffle']


class PixelShuffle(nn.PixelShuffle, Module):
    """Invertible pixelshuffle layer"""
    reversible = True
    forward = Module.forward

    def __init__(self, upscale_factor, channels_first=True):
        super().__init__(upscale_factor)
        self.channels_first = channels_first  # C * r * r instead of r * r * C

    def function(self, inputs):  # pylint: disable=arguments-differ
        stride = self.get_stride(inputs.dim() - 2)
        if all(s < 2 for s in stride):
            return inputs
        dims = tuple(d * s for d, s in zip(inputs.shape[2:], stride))
        inputs = self.shuffle(self.chunk_channels(inputs))
        inputs = inputs.reshape(inputs.shape[0], -1, *dims)
        return inputs

    def inverse(self, outputs):  # pylint: disable=arguments-differ
        stride = self.get_stride(outputs.dim() - 2)
        if all(s < 2 for s in stride):
            return outputs
        outputs = self.shuffle(self.chunk_dims(outputs), inverse=True)
        outputs = outputs.contiguous().flatten(1, -len(stride) - 1)
        return outputs

    def chunk_channels(self, inputs):
        """Chunk tensor's channels [C * r * r, H, W] -> [C, r, r, H, W]"""
        stride = list(self.get_stride(inputs.dim() - 2))
        channels = [-1] + stride if self.channels_first else stride + [-1]
        return inputs.view(inputs.shape[0], *channels, *inputs.shape[2:])

    def chunk_dims(self, inputs):
        """Chunk tensor's dimensions [C, H, W] -> [C, H / r, r, W / r, r]"""
        stride = self.get_stride(inputs.dim() - 2)
        dims = tuple((d // s, s) for d, s in zip(inputs.shape[2:], stride))
        return inputs.view(*itertools.chain(inputs.shape[:2], *dims))

    def shuffle(self, inputs, inverse=False):
        """Order chunked tensor [C, r, r, H, W] -> [C, H, r, W, r]"""
        output_dims = range(inputs.dim())
        dims, blocks = output_dims[2::2], output_dims[3::2]
        if self.channels_first:
            dims = (0, 1, *blocks, *dims)
        else:
            dims = (0, *blocks, 1, *dims)
        if not inverse:
            dims = (i for i, _ in sorted(enumerate(dims), key=lambda x: x[1]))
        return inputs.permute(*dims)

    def get_stride(self, dim=1):
        """Get the stride as a tuple"""
        stride = self.upscale_factor
        if isinstance(stride, int):
            return (stride, ) * dim
        assert isinstance(stride, tuple)
        return stride

    def extra_repr(self):
        return f'{super().extra_repr()}, {Module.extra_repr(self)}'
