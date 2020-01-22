from typing import Tuple, List

import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

from nsynth.decoder import WavenetDecoder
from nsynth.encoder import TemporalEncoder
from nsynth.functional import shift1d
from nsynth.modules import AutoEncoder, VQEmbedding
from nsynth.utils import clean_init_args
from .functional import destroy_along_axis


class WavenetVAE(AutoEncoder):
    def __init__(self, in_channels: int, out_channels: int, latent_width: int,
                 encoder_width: int, decoder_width: int, n_blocks: int = 3,
                 n_layers: int = 10):
        super(WavenetVAE, self).__init__()

        self.latent_width = latent_width
        self.encoder_params = dict(in_channels=in_channels,
                                   out_channels=2 * latent_width,
                                   n_blocks=n_blocks, n_layers=n_layers,
                                   width=encoder_width)
        self.decoder_params = dict(in_channels=in_channels,
                                   out_channels=out_channels,
                                   n_blocks=n_blocks, n_layers=n_layers,
                                   residual_width=2 * decoder_width,
                                   skip_width=decoder_width,
                                   conditional_dims=[(latent_width, 1)])

    def _latent(self, embedding: torch.Tensor):
        q_μ = embedding[:, :self.latent_width, :]
        q_σ = F.softplus(embedding[:, self.latent_width:, :]) + 1e-7

        q_z = dist.Normal(q_μ, q_σ)
        z = q_z.rsample()
        log_q_z = q_z.log_prob(z)
        return z, log_q_z


class WavenetMultiVAE(WavenetVAE):
    def __init__(self, n: int, *args, **kwargs):
        super(WavenetMultiVAE, self).__init__(*args, **kwargs)
        self.params = clean_init_args(locals().copy())

        self.encoder = TemporalEncoder(conditional_dims=[self.latent_width],
                                       **self.encoder_params)

        self.decoder_params['conditional_dims'] = [(self.latent_width, 1),
                                                   self.latent_width]
        self.decoders = nn.ModuleList(
            [WavenetDecoder(**self.decoder_params) for _ in range(n)])

        self.z_c = None

    def forward(self, x: torch.Tensor, offsets: torch.Tensor) \
            -> Tuple[torch.Tensor, dist.Normal, torch.Tensor]:
        if self.z_c is None:
            self.z_c = torch.zeros((x.shape[0], self.latent_width),
                                   device=x.device, dtype=torch.float32)
        self.z_c[offsets == 0, :] = 0  # Reset conds for mix at the start
        q_μ_σ = self.encoder(x, [self.z_c])
        z, log_q_z = self._latent(q_μ_σ)
        y_tilde = self._decode(x, [z, self.z_c])
        # I am detaching the z[t-1] right?
        self.z_c = z[:, :, -1].detach()
        return y_tilde, z, log_q_z

    def test_forward(self, x: torch.Tensor, destroy: float = 0) \
            -> torch.Tensor:
        embedding = self.encoder(x)
        q_μ = embedding[:, :self.latent_width, :]
        if destroy > 0:
            q_μ = destroy_along_axis(q_μ, destroy)

        log_x_t = self._decode(x, q_μ)
        return log_x_t

    def _decode(self, x: torch.Tensor, z: List[torch.Tensor]) \
            -> torch.Tensor:
        x = shift1d(x, -1)
        logits = [dec(x, z) for dec in self.decoders]
        return torch.cat(logits, dim=1)


class ConditionalWavenetVQVAE(nn.Module):
    def __init__(self, n_sources: int, K: int = 1, D: int = 512,
                 n_blocks: int = 3, n_layers: int = 10,
                 encoder_width: int = 256, decoder_width: int = 256,
                 in_channels: int = 1, out_channels: int = 256,
                 device: str = 'cpu', ):
        super(ConditionalWavenetVQVAE, self).__init__()
        self.params = clean_init_args(locals().copy())

        self.device = device
        self.n_sources = n_sources
        self.encoder_params = dict(in_channels=in_channels, out_channels=D,
                                   n_blocks=n_blocks, n_layers=n_layers,
                                   width=encoder_width, device=device,
                                   conditional_dims=[n_sources])

        self.decoder_params = dict(in_channels=in_channels,
                                   out_channels=out_channels,
                                   conditional_dims=[(D, 1)],
                                   n_blocks=n_blocks, n_layers=n_layers,
                                   skip_width=decoder_width,
                                   residual_width=2 * decoder_width)

        self.encoder = TemporalEncoder(**self.encoder_params)

        self.decoder = WavenetDecoder(**self.decoder_params)
        self.codebook = VQEmbedding(K, D)

    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c_l = F.one_hot(labels, self.n_sources).float().to(x.device)
        z = self.encoder(x, [c_l])
        return z

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        z = self.encode(x, labels)
        z_q_x_st, z_q_x = self.codebook.straight_through(z)
        x_tilde = self.decoder(x, [z_q_x_st])
        return x_tilde, z, z_q_x

    def test_forward(self, x: torch.Tensor, labels: torch.Tensor,
                     destroy: float = 0):
        z = self.encode(x, labels)
        if destroy > 0:
            z = destroy_along_axis(z, destroy)
        x_tilde = self.decoder(x, [z])
        return x_tilde
