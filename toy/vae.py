from typing import Tuple

import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

from nsynth.decoder import WaveNetDecoder, WavenetDecoder
from nsynth.encoder import TemporalEncoder, ConditionalTemporalEncoder
from nsynth.functional import shift1d
from nsynth.modules import AutoEncoder, VQEmbedding
from .functional import destroy_along_axis


class WavenetVAE(AutoEncoder):
    def __init__(self, bottleneck_dims: int, encoder_width: int,
                 decoder_width: int, n_layers: int = 10, n_blocks: int = 3,
                 quantization_channels: int = 256,
                 channels: int = 1, gen: bool = False):
        super(WavenetVAE, self).__init__()
        self.bottleneck_dims = bottleneck_dims
        self.encoder_params = dict(in_channels=channels,
                                   out_channels=2 * bottleneck_dims,
                                   n_blocks=n_blocks, n_layers=n_layers,
                                   width=encoder_width)

        self.decoder_params = dict(bottleneck_dims=bottleneck_dims,
                                   channels=channels, width=decoder_width,
                                   n_layers=n_layers, n_blocks=n_blocks,
                                   quantization_channels=quantization_channels,
                                   gen=gen)

    def _latent(self, embedding: torch.Tensor):
        q_loc = embedding[:, :self.bottleneck_dims, :]
        q_scale = F.softplus(embedding[:, self.bottleneck_dims:, :]) + 1e-7

        q = dist.Normal(q_loc, q_scale)
        x_q = q.rsample()
        x_q_log_prob = q.log_prob(x_q)
        return x_q, x_q_log_prob


class WavenetMultiVAE(WavenetVAE):
    def __init__(self, n: int, *args, **kwargs):
        super(WavenetMultiVAE, self).__init__(*args, **kwargs)

        self.encoder = TemporalEncoder(**self.encoder_params)
        self.decoders = nn.ModuleList(
            [WaveNetDecoder(**self.decoder_params) for _ in range(n)])

    def forward(self, x: torch.Tensor) \
            -> Tuple[torch.Tensor, dist.Normal, torch.Tensor]:
        embedding = self.encoder(x)
        x_q, x_q_log_prob = self._latent(embedding)
        logits = self._decode(x, x_q)
        return logits, x_q, x_q_log_prob

    def test_forward(self, x: torch.Tensor, destroy: float = 0) -> torch.Tensor:
        embedding = self.encoder(x)
        q_loc = embedding[:, :self.bottleneck_dims, :]
        if destroy > 0:
            q_loc = destroy_along_axis(q_loc, destroy)

        logits = self._decode(x, q_loc)
        return logits

    def _decode(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        x = shift1d(x, -1)
        logits = [dec(x, embedding) for dec in self.decoders]
        return torch.cat(logits, dim=1)


class ConditionalWavenetVAE(WavenetVAE):
    def __init__(self, n: int, *args, device: str = 'cpu', **kwargs):
        super(ConditionalWavenetVAE, self).__init__(*args, **kwargs)
        self.encoder = ConditionalTemporalEncoder(n_classes=n, device=device,
                                                  **self.encoder_params)
        self.decoder = WaveNetDecoder(**self.decoder_params)
        self.n, self.device = n, device

    def _condition(self, labels: torch.Tensor) -> torch.Tensor:
        return F.one_hot(labels, self.n).float().to(self.device)

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        embedding = self.encoder(x, self._condition(labels))
        x_q, x_q_log_prob = self._latent(embedding)
        logits = self._decode(x, x_q)

        return logits, x_q, x_q_log_prob

    def test_forward(self, x: torch.Tensor, labels: torch.Tensor,
                     destroy: float = 0):
        embedding = self.encoder(x, self._condition(labels))
        q_loc = embedding[:, :self.bottleneck_dims, :]
        if destroy > 0:
            q_loc = destroy_along_axis(q_loc, destroy)

        logits = self._decode(x, q_loc)
        return logits

    def _decode(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        x = shift1d(x, -1)
        logits = self.decoder(x, embedding)
        return logits


class ConditionalWavenetVQVAE(nn.Module):
    def __init__(self, n_sources: int, K: int = 1, D: int = 512,
                 n_blocks: int = 3, n_layers: int = 10,
                 encoder_width: int = 256, decoder_width: int = 256,
                 in_channels: int = 1, out_channels: int = 256,
                 device: str = 'cpu', ):
        super(ConditionalWavenetVQVAE, self).__init__()
        self.device = device
        self.n_sources = n_sources
        self.encoder_params = dict(in_channels=in_channels, out_channels=D,
                                   n_blocks=n_blocks, n_layers=n_layers,
                                   width=encoder_width, n_classes=n_sources,
                                   device=device)

        self.decoder_params = dict(in_channels=in_channels,
                                   out_channels=out_channels,
                                   conditional_dims=[(D, False),
                                                     (n_sources, True)],
                                   n_blocks=n_blocks, n_layers=n_layers,
                                   skip_width=decoder_width,
                                   residual_width=2 * decoder_width)

        self.encoder = ConditionalTemporalEncoder(**self.encoder_params)

        self.decoder = WavenetDecoder(**self.decoder_params)
        self.codebook = VQEmbedding(K, D)

    def _condition(self, labels: torch.Tensor) -> torch.Tensor:
        return F.one_hot(labels, self.n_sources).float().to(self.device)

    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        source_cond = self._condition(labels)
        embedding = self.encoder(x, source_cond)
        latents = self.codebook(embedding)
        return latents

    def decode(self, x: torch.Tensor, embedding: torch.Tensor,
               labels: torch.Tensor) -> torch.Tensor:
        # (B, D, H, W)
        source_cond = self._condition(labels)
        z_q_x = self.codebook.embedding(embedding).permute(0, 2, 1)
        x_tilde = self.decoder(x, [z_q_x, source_cond])
        return x_tilde

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        source_cond = self._condition(labels)
        embedding = self.encoder(x, source_cond)
        z_q_x_st, z_q_x = self.codebook.straight_through(embedding)
        x_tilde = self.decoder(x, [z_q_x_st, source_cond])
        return x_tilde, embedding, z_q_x

    def test_forward(self, x: torch.Tensor, labels: torch.Tensor,
                     destroy: float = 0):
        source_cond = self._condition(labels)
        embedding = self.encoder(x, source_cond)
        if destroy > 0:
            embedding = destroy_along_axis(embedding, destroy)
        x_tilde = self.decoder(x, [embedding, source_cond])
        return x_tilde
