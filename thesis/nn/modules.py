from typing import List

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchaudio.transforms import MelSpectrogram as _MelSpectrogram


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        causal=False,
        bias=True,
        groups=1,
    ):
        super(Conv1d, self).__init__()

        self.causal = causal
        if self.causal:
            self.padding = dilation * (kernel_size - 1)
        else:
            self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
            bias=bias,
            groups=groups,
        )
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, tensor):
        out = self.conv(tensor)
        if self.causal and self.padding != 0:
            out = out[:, :, : -self.padding]
        return out


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, groups=1):
        super(ZeroConv1d, self).__init__()

        self.conv = nn.Conv1d(in_channel, out_channel, 1, padding=0, groups=groups)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1), requires_grad=True)

    def forward(self, x):
        out = self.conv(x)
        out = out * torch.exp(self.scale * 3)
        return out


class LinearInvert(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super(LinearInvert, self).__init__(
            in_features=in_features, out_features=out_features, bias=True
        )
        self.inv_weights = None
        # I cannot init the weights to be orthonormal as they're not gonna be
        # square. :((((((
        self.weight.normal_()
        self.bias.normal_()

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if reverse:
            if self.inv_weights is None:
                inv_w = self.weight.type(x.dtype).inverse()
                self.inv_weights = Variable(inv_w.unsqueeze(-1))
            y = F.linear(x - self.bias, self.inv_weights, None)
            return y
        else:
            y = super(LinearInvert, self).forward(x)
            return y


class STFTUpsample(nn.Module):
    def __init__(self, kernel_sizes: List[int]):
        super(STFTUpsample, self).__init__()

        self.up = nn.Sequential()
        for i, ks in enumerate(kernel_sizes):
            conv = nn.ConvTranspose2d(
                1, 1, (3, 2 * ks), padding=(1, ks // 2), stride=(1, ks)
            )
            conv = nn.utils.weight_norm(conv)
            nn.init.kaiming_normal_(conv.weight)
            self.up.add_module(f"c{i}", conv)
            self.up.add_module(f"a{i}", nn.LeakyReLU(0.4))

    def forward(self, c: torch.Tensor, width: int):
        c = self.up(c.unsqueeze(1)).squeeze(1)
        si = (c.shape[-1] - width) // 2
        c = c[..., si : si + width]
        return c


class MelSpectrogram(_MelSpectrogram):
    def __init__(self):
        super(MelSpectrogram, self).__init__(
            sample_rate=16000,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            f_min=125,
            f_max=7600,
        )
        self.reference = 20.0
        self.min_db = -100.0

    def forward(self, waveform):
        mel_specgram = super(MelSpectrogram, self).forward(waveform)
        mel_spectrogram = (
            20 * torch.log10(mel_specgram.clamp(min=1e-4)) - self.reference
        )
        mel_spectrogram = (mel_spectrogram - self.min_db) / (-self.min_db)
        # mel_spectrogram = ((mel_spectrogram - self.min_db) / (-self.min_db)).clamp(0, 1)
        return mel_spectrogram
