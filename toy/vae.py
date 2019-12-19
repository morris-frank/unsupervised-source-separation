from typing import Tuple
import torch
from torch import nn
from torch import distributions as dist
from torch.nn import functional as F

from nsynth.decoder import WaveNetDecoder
from nsynth.encoder import TemporalEncoder
from nsynth.functional import shift1d
from nsynth.modules import AutoEncoder


class WavenetMultiVAE(AutoEncoder):
    def __init__(self, n: int, bottleneck_dims: int, encoder_width: int,
                 decoder_width: int, n_layers: int = 10, n_blocks: int = 3,
                 quantization_channels: int = 256,
                 channels: int = 1, gen: bool = False):
        """
        :param n: number of decoders
        :param bottleneck_dims: Number of dims in the latent bottleneck.
        :param encoder_width: Width of the hidden layers in the encoder (Non-
            causal Temporal encoder).
        :param decoder_width: Width of the hidden layers in the decoder
            (WaveNet).
        :param n_layers: number of layers in each block of encoder and decoder
        :param n_blocks: number of blocks for both
        :param quantization_channels:
        :param channels: Number of input channels.
        :param gen: Is this generation ?
        """
        super(WavenetMultiVAE, self).__init__()
        self.encoder = TemporalEncoder(bottleneck_dims=2 * bottleneck_dims,
                                       channels=channels, width=encoder_width,
                                       n_layers=n_layers, n_blocks=n_blocks)

        decoder_args = dict(bottleneck_dims=bottleneck_dims,
                            channels=channels, width=decoder_width,
                            n_layers=n_layers, n_blocks=n_blocks,
                            quantization_channels=quantization_channels,
                            gen=gen)
        self.decoders = nn.ModuleList(
            [WaveNetDecoder(**decoder_args) for _ in range(n)])

    def forward(self, x: torch.Tensor) \
            -> Tuple[torch.Tensor, dist.Normal, torch.Tensor]:
        embedding = self.encoder(x)

        q_loc = embedding[:, :self.bottleneck_dims, :]
        q_scale = F.softplus(embedding[:, self.bottleneck_dims:, :]) + 1e-7

        q = dist.Normal(q_loc, q_scale)
        x_q = q.rsample()
        x_q_log_prob = q.log_prob(x_q)

        x = shift1d(x, -1)
        logits = [dec(x, x_q) for dec in self.decoders]
        return logits, x_q, x_q_log_prob

