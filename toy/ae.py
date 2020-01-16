import torch
from torch import nn

from nsynth.decoder import WavenetDecoder
from nsynth.encoder import TemporalEncoder
from nsynth.functional import shift1d
from nsynth.modules import AutoEncoder
from .functional import destroy_along_axis


class WavenetMultiAE(AutoEncoder):
    """
    The complete WaveNetAutoEncoder model.
    """

    def __init__(self, n: int, in_channels: int, out_channels: int,
                 latent_width: int, encoder_width: int, decoder_width: int,
                 n_blocks: int = 3, n_layers: int = 10):
        """
        :param n: number of decoders
        :param latent_width: Number of dims in the latent bottleneck.
        :param encoder_width: Width of the hidden layers in the encoder (Non-
            causal Temporal encoder).
        :param decoder_width: Width of the hidden layers in the decoder
            (WaveNet).
        :param n_layers: number of layers in each block of encoder and decoder
        :param n_blocks: number of blocks for both
        :param in_channels: Number of input in_channels.
        :param out_channels:
        """
        super(WavenetMultiAE, self).__init__()
        self.args = locals().copy()
        del self.args['self']

        self.latent_width = latent_width
        self.encoder = TemporalEncoder(
            in_channels=in_channels, out_channels=latent_width,
            n_blocks=n_blocks, n_layers=n_layers, width=encoder_width
        )

        decoder_args = dict(in_channels=in_channels,
                            out_channels=out_channels,
                            n_blocks=n_blocks, n_layers=n_layers,
                            residual_width=2 * decoder_width,
                            skip_width=decoder_width,
                            conditional_dims=[(latent_width, False)])
        self.decoders = nn.ModuleList(
            [WavenetDecoder(**decoder_args) for _ in range(n)])

    def _decode(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        x = shift1d(x, -1)
        logits = [dec(x, [embedding]) for dec in self.decoders]
        return torch.cat(logits, dim=1)

    def test_forward(self, x: torch.Tensor, destroy: float = 0) -> torch.Tensor:
        embedding = self.encoder(x)
        embedding = destroy_along_axis(embedding, destroy)
        return self._decode(x, embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        return self._decode(x, embedding)
