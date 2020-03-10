from typing import Tuple

import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F
from torchaudio.transforms import MelSpectrogram

from . import BaseModel
from ..wavenet import Wavenet
from ...utils import clean_init_args


class q_sǀm(nn.Module):
    def __init__(self, n_classes, mel_channels, with_head=True):
        super(q_sǀm, self).__init__()
        head_dim = 128
        arm_dim = 32 if with_head else 64
        self.with_head = with_head

        if with_head:
            self.f = Wavenet(
                in_channels=1,
                out_channels=head_dim,
                n_blocks=1,
                n_layers=11,
                residual_channels=head_dim,
                gate_channels=head_dim,
                skip_channels=head_dim,
                cin_channels=mel_channels,
            )

        self.f_k = nn.ModuleList()
        self.f_k_α = nn.ModuleList()
        self.f_k_β = nn.ModuleList()
        for k in range(n_classes):
            self.f_k.append(
                Wavenet(
                    in_channels=head_dim if with_head else 1,
                    out_channels=arm_dim,
                    n_blocks=2,
                    n_layers=11,
                    residual_channels=arm_dim,
                    gate_channels=arm_dim,
                    skip_channels=arm_dim,
                    cin_channels=mel_channels,
                )
            )
            self.f_k_α.append(nn.Sequential(nn.Conv1d(arm_dim, 1, 1), nn.Softplus()))
            self.f_k_β.append(nn.Sequential(nn.Conv1d(arm_dim, 1, 1), nn.Softplus()))

    def forward(self, m: torch.Tensor, m_mel: torch.Tensor):
        α, β = [], []

        f_m = self.f(m, m_mel) if self.with_head else m
        for k in range(len(self.f_k)):
            f_k = self.f_k[k](f_m, m_mel)
            α.append(self.f_k_μ[k](f_k) + 1e-7)
            β.append(self.f_k_σ[k](f_k) + 1e-7)
        return torch.cat(α, dim=1), torch.cat(β, dim=1)


class UMixer(BaseModel):
    def __init__(self, mel_channels: int = 80):
        super(UMixer, self).__init__()
        self.params = clean_init_args(locals().copy())

        self.n_classes = 4

        self.q_sǀm = q_sǀm(self.n_classes, mel_channels)

        self.p_s = None

        self.p_mǀs = Wavenet(
            in_channels=self.n_classes,
            out_channels=1,
            n_blocks=1,
            n_layers=8,
            residual_channels=32,
            gate_channels=32,
            skip_channels=32,
            cin_channels=None,
        )

        self.upsample_conv = nn.ModuleList()
        for s in [16, 16]:
            convt = nn.ConvTranspose2d(
                1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s)
            )
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(0.4))

        n_fft = 1024
        hop_length = 256
        sr = 16000
        self.mel = MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=mel_channels,
            f_min=125,
            f_max=7600,
        )

    def forward(
        self, m: torch.Tensor, m_mel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        m_mel = self.upsample(m_mel)
        m_mel = m_mel[:, :, : m.shape[-1]]

        α, β = self.q_sǀm(m, m_mel)

        q_s = dist.Beta(α, β)
        ŝ = q_s.rsample()
        log_q_ŝ = q_s.log_prob(ŝ)

        for k in range(self.n_classes):
            # Get Log likelihood under prior
            ŝ_mel = self.upsample(self.mel(ŝ[:, k, :]))
            with torch.no_grad():
                log_p_ŝ, _ = self.p_s[k](ŝ[:, None, k, :], ŝ_mel, sum_log_p=False)
                log_p_ŝ = log_p_ŝ.detach()[:, None]

            # Kullback Leibler for this k'th source
            KL_k = -torch.mean(log_p_ŝ - log_q_ŝ[:, k, :])
            setattr(self.ℒ, f"KL_{k}", KL_k)

        m_ = self.p_mǀs(ŝ)
        self.ℒ.l1_recon = F.l1_loss(m_, m)

        return ŝ, m_

    def test(self, m: torch.Tensor, m_mel: torch.Tensor) -> torch.Tensor:
        β = 1.1
        _ = self.forward(m, m_mel)

        ℒ = self.ℒ.l1_recon

        for k in range(self.n_classes):
            ℒ += β * getattr(self.ℒ, f"KL_{k}")

        return ℒ

    def upsample(self, c):
        c = c.unsqueeze(1)
        for f in self.upsample_conv:
            c = f(c)
        c = c.squeeze(1)
        return c
