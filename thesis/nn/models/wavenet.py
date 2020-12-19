from torch import nn
from . import BaseModel
from ..wavenet import Wavenet as WaveNetModule
from ...utils import clean_init_args
from ...audio import encode_μ_law
from torch.nn import functional as F


class WaveNet(BaseModel):
    def __init__(self, in_channels, **kwargs):
        super(WaveNet, self).__init__(**kwargs)
        self.params = clean_init_args(locals().copy())

        self.in_channels = in_channels
        self.out_channels = 256

        self.net = nn.Sequential(
            WaveNetModule(in_channels=in_channels,
                          out_channels=in_channels * self.out_channels,
                          cin_channels=None,
                          n_blocks=3,
                          n_layers=10,
                          bias=True,
                          causal=True,
                          groups=in_channels),
            nn.Sigmoid()
        )

    def forward(self, s, _ce=True):
        if _ce:
            return self.net(s)
        else:
            N, C, L = s.shape
            s = encode_μ_law(s)
            ŝ = self.net(s[..., :-1])
            ŝ = ŝ.view(N, C, self.out_channels, L-1)
            ŝ = ŝ.gather(2, s[..., 1:].long().unsqueeze(2))
            return None, ŝ

    def test(self, s):
        s = encode_μ_law(s)
        ŝ = self.forward(s[..., :-1])
        ℒ = 0
        for i in range(self.in_channels):
            setattr(self.ℒ, f"CE_{i}", F.cross_entropy(ŝ[:, i*256:(i+1)*256, :], s[:, i, 1:].long()))
            ℒ = ℒ + getattr(self.ℒ, f"CE_{i}")
        return ℒ
