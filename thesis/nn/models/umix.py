from typing import Tuple

from torchaudio.transforms import MelSpectrogram
from torch import nn
import torch
from torch.nn import functional as F

from . import BaseModel
from ..wavenet import Wavenet
from ...functional import rsample_truncated_normal
from ...utils import clean_init_args


class UMixer(BaseModel):
    def __init__(self, mel_channels: int = 80):
        super(UMixer, self).__init__()
        self.params = clean_init_args(locals().copy())

        self.n_classes = 4

        self.q_sǀm = Wavenet(
            in_channels=1,
            out_channels=2 * self.n_classes,
            n_blocks=3,
            n_layers=11,
            residual_channels=256,
            gate_channels=256,
            skip_channels=256,
            cin_channels=mel_channels,
        )

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
        self.mel = MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=mel_channels, f_min=125, f_max=7600)

    def forward(
        self, m: torch.Tensor, m_mel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        m_mel = self.upsample(m_mel)
        m_mel = m_mel[:, :, :m.shape[-1]]

        η = self.q_sǀm(m, m_mel)
        μ, σ = η.chunk(2, dim=1)

        # yes?
        μ = μ
        σ = F.softplus(σ) + 1e-7

        ŝ, log_q_ŝ = rsample_truncated_normal(μ, σ, ll=True)

        for k in range(self.n_classes):
            # Get Log likelihood under prior
            ŝ_mel = self.upsample(self.mel(ŝ[:, k, :]))
            with torch.no_grad():
                log_p_ŝ, _ = self.p_s[k](ŝ[:, None, k, :], ŝ_mel)

            # Kullback Leibler for this k'th source
            KL_k = - torch.mean(log_p_ŝ.detach()[:, None] - log_q_ŝ[:, k, :])
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
