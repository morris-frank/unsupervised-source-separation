from typing import Tuple

from random import random

import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F
from torchaudio.transforms import MelSpectrogram

from . import BaseModel
from ..wavenet import Wavenet
from ...utils import clean_init_args
from ...functional import encode_μ_law, decode_μ_law


class q_sǀm(nn.Module):
    def __init__(self, out_channels, mel_channels, dim):
        super(q_sǀm, self).__init__()
        self.f = Wavenet(
            in_channels=1,
            out_channels=out_channels,
            n_blocks=3,
            n_layers=11,
            residual_channels=dim,
            gate_channels=dim,
            skip_channels=dim,
            cin_channels=mel_channels,
        )

    def forward(self, m: torch.Tensor, m_mel: torch.Tensor):
        return F.log_softmax(self.f(m, m_mel), dim=1)


class CUMixer(BaseModel):
    def __init__(self, μ: int, mel_channels: int = 80, width: int = 64):
        super(CUMixer, self).__init__()
        self.params = clean_init_args(locals().copy())
        self.name = "cat_supervised"
        self.μ = μ

        self.n_classes = 4

        # The encoders
        self.q_sǀm = nn.ModuleList()
        for k in range(self.n_classes):
            self.q_sǀm.append(q_sǀm(μ, mel_channels, width))

        # The placeholder for the prior networks
        self.p_s = None

        # the decoder
        self.p_mǀs = Wavenet(
            in_channels=self.n_classes,
            out_channels=μ,
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

    def q_s(self, m, m_mel):
        m_mel = self.upsample(m_mel)
        m_mel = m_mel[:, :, : m.shape[-1]]

        logits = [q(m, m_mel) for q in self.q_sǀm]
        logits = torch.stack(logits, dim=1).transpose(2, 3)
        q_s = dist.Categorical(logits=logits)
        return q_s

    def forward(
        self, m: torch.Tensor, m_mel: torch.Tensor
    ) -> dist.Categorical:
        q_s = self.q_s(m, m_mel)
        ŝ = q_s.sample()
        log_q_ŝ = q_s.log_prob(ŝ)
        ŝ = decode_μ_law(ŝ, self.μ)

        for k in range(self.n_classes):
            # Get Log likelihood under prior
            ŝ_mel = self.upsample(self.mel(ŝ[:, k, :]))
            with torch.no_grad():
                log_p_ŝ, _ = self.p_s[k](ŝ[:, None, k, :], ŝ_mel)
                log_p_ŝ = log_p_ŝ.detach()[:, None]

            # Kullback Leibler for this k'th source
            KL_k = -torch.mean(log_p_ŝ - log_q_ŝ[:, k, :])
            setattr(self.ℒ, f"KL_{k}", KL_k)

        m_ = self.p_mǀs(ŝ)
        self.ℒ.ce_mix = F.cross_entropy(m_, encode_μ_law(m, self.μ).squeeze())

        return q_s

    def test(
        self, x: Tuple[torch.Tensor, torch.Tensor], s: torch.Tensor
    ) -> torch.Tensor:
        m, m_mel = x
        s = encode_μ_law(s, self.μ)
        β = 1.1
        q_s = self.forward(m, m_mel)

        ℒ = self.ℒ.ce_mix

        for k in range(self.n_classes):
            if random() < 0.1:
                logits = q_s.logits[:, k, ...].transpose(1, 2)
                setattr(self.ℒ, f"nll_{k}", F.nll_loss(logits, s[:, k, ...]))
                ℒ += getattr(self.ℒ, f"nll_{k}")

            ℒ += β * getattr(self.ℒ, f"KL_{k}")

        return ℒ

    def upsample(self, c):
        c = c.unsqueeze(1)
        for f in self.upsample_conv:
            c = f(c)
        c = c.squeeze(1)
        return c
