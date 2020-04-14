import torch
from torch.nn import functional as F

from thesis.nn.models import BaseModel
from thesis.nn.wavenet import Wavenet
from thesis.utils import clean_init_args


class Decriminator(BaseModel):
    def __init__(self, n_classes, width, mel_channels, **kwargs):
        super(Decriminator, self).__init__(**kwargs)
        self.params = clean_init_args(locals().copy())

        self.wn = Wavenet(
            in_channels=n_classes,
            out_channels=n_classes,
            n_blocks=3,
            n_layers=11,
            residual_channels=width * n_classes,
            gate_channels=width * n_classes,
            skip_channels=width * n_classes,
            cin_channels=mel_channels,
            bias=False,
            fc_kernel_size=3,
            fc_channels=2048,
            groups=n_classes,
        )

    def forward(self, s, s_mel=None):
        s_mel = F.interpolate(s_mel, s.shape[-1], mode="linear", align_corners=False)
        return self.wn(s, s_mel)

    def test(self, x, t):
        s, s_mel = x
        n, _, l = s.shape

        ik = torch.tensor([0, 1, 2, 3]).repeat(n, 1)
        t = t.repeat(4, 1).T
        tgt = (ik == t).long().unsqueeze(-1).repeat(1, 1, l).to(s.device)

        t_s = self.forward(s, s_mel)

        self.ℒ.target = F.mse_loss(t_s, tgt)

        return self.ℒ.target
