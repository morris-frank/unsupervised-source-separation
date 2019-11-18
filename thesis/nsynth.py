import torch
from torch import nn

from .modules import BlockWiseConv1d


class NCTemporalEncoder(nn.Module):
    def __init__(self,
                 n_channels: int,
                 n_layers: int = 30,
                 n_stages: int = 10,
                 hidden_dims: int = 128,
                 kernel_size: int = 3,
                 bottleneck_dims: int = 16,
                 use_bias: bool = True):
        super(NCTemporalEncoder, self).__init__()

        self.encoder = []
        self.encoder.append(
            BlockWiseConv1d(in_channels=n_channels,
                            out_channels=hidden_dims,
                            kernel_size=kernel_size,
                            causal=False,
                            block_size=1,
                            bias=use_bias)
        )
        for idx in range(n_layers):
            dilation = 2**(idx % n_stages)
            self.encoder.extend([
                nn.ReLU(),
                BlockWiseConv1d(in_channels=hidden_dims,
                                out_channels=hidden_dims,
                                kernel_size=kernel_size,
                                causal=False,
                                block_size=dilation,
                                bias=use_bias),
                nn.ReLU(),
                BlockWiseConv1d(in_channels=hidden_dims,
                                out_channels=hidden_dims,
                                kernel_size=1,
                                causal=True,
                                block_size=1,
                                bias=use_bias)
            ])

        # Bottleneck
        self.encoder.append(
            nn.Conv1d(in_channels=hidden_dims,
                      out_channels=bottleneck_dims,
                      kernel_size=1,
                      bias=use_bias)
        )
        self.encoder = nn.ModuleList(self.encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
