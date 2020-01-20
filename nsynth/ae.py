from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .decoder import WavenetDecoder
from .encoder import TemporalEncoder
from .functional import shift1d
from .modules import AutoEncoder
from .utils import clean_init_args


class WavenetAE(AutoEncoder):
    """
    The complete WaveNetAutoEncoder model.
    """

    def __init__(self,
                 in_channels: int, out_channels: int, latent_width: int,
                 encoder_width: int, decoder_width: int, n_blocks: int = 3,
                 n_layers: int = 10):
        """
        :param in_channels: Number of input in_channels.
        :param out_channels:
        :param latent_width: Number of dims in the latent bottleneck.
        :param encoder_width: Width of the hidden layers in the encoder (Non-
            causal Temporal encoder).
        :param decoder_width: Width of the hidden layers in the decoder
            (WaveNet).
        :param n_blocks: number of blocks for both
        :param n_layers: number of layers in each block of encoder and decoder
        """
        super(WavenetAE, self).__init__()
        self.params = clean_init_args(locals().copy())

        self.encoder = TemporalEncoder(
            in_channels=in_channels, out_channels=latent_width,
            n_blocks=n_blocks, n_layers=n_layers, width=encoder_width
        )
        self.decoder = WavenetDecoder(
            in_channels=in_channels, out_channels=out_channels,
            n_blocks=n_blocks, n_layers=n_layers,
            residual_width=2 * decoder_width, skip_width=decoder_width,
            conditional_dims=[(latent_width, False)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        x = shift1d(x, -1)
        logits = self.decoder(x, [embedding])
        return logits

    @staticmethod
    def loss_function(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                      device: str, progress: float) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        del progress
        logits = model(x)
        loss = F.cross_entropy(logits, y.to(device))
        return loss, logits
