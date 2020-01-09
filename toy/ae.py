import torch
from torch import nn

from nsynth.decoder import WaveNetDecoder
from nsynth.encoder import TemporalEncoder
from nsynth.functional import shift1d
from nsynth.modules import AutoEncoder
from .functional import destroy_along_axis


class WavenetMultiAE(AutoEncoder):
    """
    The complete WaveNetAutoEncoder model.
    """

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
        super(WavenetMultiAE, self).__init__()
        self.bottleneck_dims = bottleneck_dims
        self.encoder = TemporalEncoder(in_channels=channels,
                                       out_channels=bottleneck_dims,
                                       n_blocks=n_blocks, n_layers=n_layers,
                                       width=encoder_width)

        decoder_args = dict(bottleneck_dims=bottleneck_dims,
                            channels=channels, width=decoder_width,
                            n_layers=n_layers, n_blocks=n_blocks,
                            quantization_channels=quantization_channels,
                            gen=gen)
        self.decoders = nn.ModuleList(
            [WaveNetDecoder(**decoder_args) for _ in range(n)])

    def _decode(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        x = shift1d(x, -1)
        logits = [dec(x, embedding) for dec in self.decoders]
        return torch.cat(logits, dim=1)

    def test_forward(self, x: torch.Tensor, destroy: float = 0) -> torch.Tensor:
        embedding = self.encoder(x)
        embedding = destroy_along_axis(embedding, destroy)
        return self._decode(x, embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        return self._decode(x, embedding)
