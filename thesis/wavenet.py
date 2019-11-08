from typing import Union
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from .utils import prime_factorization
from .functional import dilate


class WaveNet(nn.Module):
    def __init__(self,
                 input_size: int,
                 n_blocks: int = 2,
                 n_layers: Union[int, None] = None,
                 quantization_channels: int = 256,
                 residual_dims: int = 32,
                 dilation_dims: int = 32,
                 skip_dims: int = 16,
                 kernel_size: int = 2,
                 use_conv_bias: bool = False
                 ):
        """
        :param input_size: The length of the input ⇒ will be used to calculate the dilations
        :param n_blocks: Number of blocks (2 [WaveNet])
        :param n_layers: Number of layers in block (== None ⇒ make from input size)
        :param quantization_channels: Number of outputs (⇒ size of μ-law quantization)
        :param residual_dims: Num of Channels in the Residual
        :param dilation_dims: Num of Channels in the Dilation
        :param skip_dims: Num of Channels in the Skip Connection
        :param kernel_size: Size of Conv Kernels (2 [WaveNet])
        :param use_conv_bias: Use bias in the conv layers (== False [WaveNet])
        """
        super(WaveNet, self).__init__()

        dilation_factors = prime_factorization(input_size)

        self.channels = 1  # Number of audio channels? stereo ⇒ == 2 (doesn't work though)
        self.n_layers = n_layers
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size

        # First layer audio ⇒ residual
        self.initial = nn.Conv1d(in_channels=self.channels,
                                 out_channels=residual_dims,
                                 kernel_size=1,
                                 bias=use_conv_bias)
        # Final Skip ⇒ Output
        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels=skip_dims,
                      out_channels=quantization_channels,
                      kernel_size=1,
                      bias=use_conv_bias),
            nn.ReLU(),
            nn.Conv1d(in_channels=quantization_channels,
                      out_channels=quantization_channels,
                      kernel_size=1,
                      bias=use_conv_bias)
        )
        self.filter_convs = self._make_conv_list(residual_dims, dilation_dims, kernel_size, use_conv_bias)
        self.gate_convs = self._make_conv_list(residual_dims, dilation_dims, kernel_size, use_conv_bias)
        self.residual_convs = self._make_conv_list(dilation_dims, residual_dims, 1, use_conv_bias)
        self.skip_convs = self._make_conv_list(dilation_dims, skip_dims, 1, use_conv_bias)

        # TODO: dilation queues

        # Size of the receptive field for each layer for all blocks (same in all blocks)
        self.dilations = [int(np.product(dilation_factors[:l])) for l in range(n_layers)]

    def _make_conv_list(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool) -> nn.ModuleList:
        """
        Returns an ModuleList of n_blocks × n_layers 1D Convolutions with the parameters
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param bias:
        :return:
        """
        module_list = []
        for _ in range(self.n_blocks * self.n_layers):
            module_list.append(nn.Conv1d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         bias=bias))
        return nn.ModuleList(module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)

        # the skip connection gets build up along side the other stuff so we have to init it here:
        skip = torch.tensor(0)

        for idx in range(self.n_layers * self.n_blocks):
            # 1. Dilate
            dilation = self.dilations[idx % self.n_layers]
            residual = dilate(x, dilation)

            # 2. Gated dilated convolution
            filters = self.filter_convs[idx](residual)
            filters = F.tanh(filters)
            gates = self.gate_convs[idx](residual)
            gates = F.tanh(gates)
            x = filters * gates

            # 3. Skip Connection
            # The skip connection builds up the actual output
            _skip = x
            if _skip.size(2) != 1:
                _skip = dilate(_skip, self.channels)
            _skip = self.skip_convs[idx](_skip)
            try:
                # Here we want to remove the front padding
                skip = skip[:, :, -_skip.size(2):]
            except IndexError:  # TODO correct error ? does this ever give an error???
                skip = torch.tensor(0)
            skip += _skip

            # 4. Prepare next input by merging with residual
            x = self.residual_convs[idx](x)
            x += residual[:, :, (self.kernel_size - 1):]

        x = self.final(skip)
        return x
