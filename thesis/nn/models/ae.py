from typing import Callable

import torch
from torch import nn

from .temporal_encoder import TemporalEncoder
from .wavenet import Wavenet
from ..optim import multi_cross_entropy
from ...functional import shift1d, destroy_along_channels
from ...utils import clean_init_args


class WavenetAE(nn.Module):
    """
    The complete WaveNetAutoEncoder model.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 latent_width: int, encoder_width: int, decoder_width: int,
                 n_decoders: int = 1, n_blocks: int = 3, n_layers: int = 10):
        """

        Args:
            in_channels: Channels of the input
            out_channels: Channels that the output shall have
            latent_width: Number of dims in the latent bottleneck
            encoder_width: Width of the hidden layers in the encoder (Non-
                causal Temporal encoder).
            decoder_width: Width of the hidden layers in the decoder
                (Wavenet).
            n_decoders: Number of WaveNet decoders
            n_blocks: Number of blocks for both decoder and encoder
            n_layers: Number of layers in each of those blocks
        """
        super(WavenetAE, self).__init__()
        self.params = clean_init_args(locals().copy())

        self.latent_width = latent_width
        self.out_channels = out_channels

        self.encoder = TemporalEncoder(
            in_channels=in_channels, out_channels=latent_width,
            n_blocks=n_blocks, n_layers=n_layers, width=encoder_width
        )

        decoder_args = dict(in_channels=in_channels,
                            out_channels=out_channels,
                            c_channels=latent_width,
                            n_blocks=n_blocks, n_layers=n_layers,
                            residual_width=2 * decoder_width,
                            skip_width=decoder_width)
        self.decoders = nn.ModuleList(
            [Wavenet(**decoder_args) for _ in range(n_decoders)])

    def _decode(self, x: torch.Tensor, embedding: torch.Tensor) \
            -> torch.Tensor:
        x = shift1d(x, -1)
        logits = [dec(x, embedding) for dec in self.decoders]
        return torch.cat(logits, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        return self._decode(x, embedding)

    def infer(self, x: torch.Tensor, destroy: float = 0) \
            -> torch.Tensor:
        embedding = self.encoder(x)
        embedding = destroy_along_channels(embedding, destroy)
        return self._decode(x, embedding)

    def loss(self) -> Callable:
        def func(model, x, y, progress):
            _ = progress  # Ignore arg
            y_tilde = model(x)
            loss = multi_cross_entropy(y_tilde, y, len(self.decoders),
                                       self.out_channels)
            return loss

        return func
