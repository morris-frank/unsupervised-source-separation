from typing import Tuple, List, Callable

import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

from . import TemporalEncoder, Wavenet
from ..modules import VQEmbedding
from ..optim import sample_kl, multi_cross_entropy
from ...functional import shift1d, destroy_along_channels
from ...utils import clean_init_args


class WavenetVAE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, latent_width: int,
                 encoder_width: int, decoder_width: int, n_decoders: int = 1,
                 n_blocks: int = 3, n_layers: int = 10):
        super(WavenetVAE, self).__init__()
        self.params = clean_init_args(locals().copy())

        self.out_channels = out_channels
        self.latent_width = latent_width

        self.encoder = TemporalEncoder(in_channels=in_channels,
                                       out_channels=2 * latent_width,
                                       n_blocks=n_blocks, n_layers=n_layers,
                                       width=encoder_width,
                                       conditional_dims=[self.latent_width])

        self.decoder_params = dict(in_channels=in_channels,
                                   out_channels=out_channels,
                                   n_blocks=n_blocks, n_layers=n_layers,
                                   residual_width=2 * decoder_width,
                                   skip_width=decoder_width,
                                   conditional_dims=[(self.latent_width, 1),
                                                     self.latent_width])
        self.decoders = nn.ModuleList(
            [Wavenet(**self.decoder_params) for _ in range(n_decoders)])

        self.z_c = None

    def _latent(self, embedding: torch.Tensor):
        q_μ = embedding[:, :self.latent_width, :]
        q_σ = F.softplus(embedding[:, self.latent_width:, :]) + 1e-7

        q_z = dist.Normal(q_μ, q_σ)
        z = q_z.rsample()
        log_q_z = q_z.log_prob(z)
        return z, log_q_z

    def forward(self, x: torch.Tensor, offsets: torch.Tensor) \
            -> Tuple[torch.Tensor, dist.Normal, torch.Tensor]:
        q_μ_σ = self.encode(x, offsets)
        z, log_q_z = self._latent(q_μ_σ)
        y_tilde = self.decode(x, [z, self.z_c])
        # I am detaching the z[t-1] right?
        self.z_c = z[:, :, -1].detach()
        return y_tilde, z, log_q_z

    def test_forward(self, x: torch.Tensor, destroy: float = 0) \
            -> torch.Tensor:
        x, offsets = x
        q_μ = self.encode(x, offsets)[:, :self.latent_width, :]
        if destroy > 0:
            q_μ = destroy_along_channels(q_μ, destroy)

        log_x_t = self.decode(x, [q_μ, self.z_c])
        # I am detaching the z[t-1] right?
        self.z_c = q_μ[:, :, -1].detach()
        return log_x_t

    def encode(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        if self.z_c is None:
            self.z_c = torch.zeros((x.shape[0], self.latent_width),
                                   device=x.device, dtype=torch.float32)
        print(f'resetted {offsets == 0}')
        self.z_c[offsets == 0, :] = 0  # Reset conds for mix at the start
        q_μ_σ = self.encoder(x, [self.z_c])
        return q_μ_σ

    def decode(self, x: torch.Tensor, z: List[torch.Tensor]) \
            -> torch.Tensor:
        x = shift1d(x, -1)
        logits = [dec(x, z) for dec in self.decoders]
        return torch.cat(logits, dim=1)

    def loss(self, β: float = 1., dβ: float = 1 / 3) \
            -> Callable:
        def func(model, x, y, progress):
            mixes, offsets = x
            logits, x_q, x_q_log_prob = model(mixes, offsets)

            # First Cross-Entropy
            ce_x = multi_cross_entropy(logits, y, len(self.decoders),
                                       self.out_channels)

            # Then Kullback-Leibler
            kl_zx = sample_kl(x_q, x_q_log_prob)

            it_β = β if dβ == 0 else β * min(progress / dβ, 1)

            loss = ce_x - it_β * kl_zx
            return loss

        return func


class ConditionalWavenetVQVAE(nn.Module):
    def __init__(self, n_sources: int, K: int = 1, D: int = 512,
                 n_blocks: int = 3, n_layers: int = 10,
                 encoder_width: int = 256, decoder_width: int = 256,
                 in_channels: int = 1, out_channels: int = 256):
        super(ConditionalWavenetVQVAE, self).__init__()
        self.params = clean_init_args(locals().copy())

        self.n_sources = n_sources

        self.encoder = TemporalEncoder(in_channels=in_channels, out_channels=D,
                                       n_blocks=n_blocks, n_layers=n_layers,
                                       width=encoder_width,
                                       conditional_dims=[n_sources])

        self.decoder = Wavenet(in_channels=in_channels,
                               out_channels=out_channels,
                               conditional_dims=[(D, 1)],
                               n_blocks=n_blocks, n_layers=n_layers,
                               skip_width=decoder_width,
                               residual_width=2 * decoder_width)
        self.codebook = VQEmbedding(K, D)

    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c_l = F.one_hot(labels, self.n_sources).float().to(x.device)
        z = self.encoder(x, [c_l])
        return z

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        z = self.encode(x, labels)
        z_q_x_st, z_q_x = self.codebook.straight_through(z)
        x = shift1d(x, -1)
        x_tilde = self.decoder(x, [z_q_x_st])
        return x_tilde, z, z_q_x

    def test_forward(self, x: torch.Tensor, labels: torch.Tensor,
                     destroy: float = 0):
        z = self.encode(x, labels)
        if destroy > 0:
            z = destroy_along_channels(z, destroy)
        x = shift1d(x, -1)
        x_tilde = self.decoder(x, [z])
        return x_tilde

    @staticmethod
    def loss(β: float = 1.1) -> Callable:
        def func(model, x, y, progress):
            _ = progress
            mixes, labels = x
            x_tilde, z_e_x, z_q_x = model(mixes, labels)
            y = y[:, 0, :].to(x_tilde.device)

            # Reconstruction loss
            loss_recons = F.cross_entropy(x_tilde, y)

            # Vector quantization objective
            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())

            # Commitment objective
            loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

            loss = loss_recons + loss_vq + β * loss_commit
            return loss

        return func
