import torch
from torch.nn import functional as F

from . import BaseModel
from ..modules import VQEmbedding
from ..temporal_encoder import TemporalEncoder
from ..wavenet import Wavenet
from ...functional import shift1d, destroy_along_channels
from ...utils import clean_init_args


class VQVAE(BaseModel):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 256,
        K: int = 1,
        D: int = 128,
        n_blocks: int = 1,
        n_layers: int = 10,
        encoder_width: int = 64,
        decoder_width: int = 64,
    ):
        super(VQVAE, self).__init__()
        self.params = clean_init_args(locals().copy())
        self.out_channels = out_channels

        self.z_eǀs = TemporalEncoder(
            in_channels=in_channels,
            out_channels=D,
            n_blocks=n_blocks,
            n_layers=n_layers,
            width=encoder_width,
        )
        self.q_zǀs = VQEmbedding(K, D)
        self.p_sǀzq = Wavenet(
            in_channels=D,
            out_channels=out_channels,
            n_blocks=n_blocks,
            n_layers=n_layers,
            skip_width=decoder_width,
            residual_width=2 * decoder_width,
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        z_e = self.z_eǀs(s)
        z_q_st, z_q = self.q_zǀs.straight_through(z_e)
        s_tilde = self.p_sǀzq(z_q_st)

        # Vector quantization objective
        self.ℒ.vec_quant = F.mse_loss(z_q, z_e.detach())
        # Commitment loss
        self.ℒ.commitment = F.mse_loss(z_e, z_q.detach())

        return s_tilde

    def infer(self, s):
        hμ = (self.out_channels - 1) // 2
        s_ = (s.float() - hμ) / hμ
        return self(s_)

    def test(self, s: torch.Tensor, *args) -> torch.Tensor:
        β = 1.1
        hμ = (self.out_channels - 1) // 2
        s_ = (s.float() - hμ) / hμ
        s_tilde = self.forward(s_)

        # Reconstruction loss
        self.ℒ.log_p_sǀzq = F.cross_entropy(s_tilde, s.squeeze())
        ℒ = self.ℒ.log_p_sǀzq + self.ℒ.vec_quant + β * self.ℒ.commitment
        return ℒ


class CVQVAE(BaseModel):
    def __init__(
        self,
        n_classes: int,
        K: int = 1,
        D: int = 512,
        n_blocks: int = 3,
        n_layers: int = 10,
        encoder_width: int = 256,
        decoder_width: int = 256,
        in_channels: int = 1,
        out_channels: int = 256,
    ):
        super(CVQVAE, self).__init__()
        self.params = clean_init_args(locals().copy())

        self.n_classes = n_classes

        self.encoder = TemporalEncoder(
            in_channels=in_channels,
            out_channels=D,
            n_blocks=n_blocks,
            n_layers=n_layers,
            width=encoder_width,
            conditional_dims=[n_classes],
        )

        self.decoder = Wavenet(
            in_channels=in_channels,
            out_channels=out_channels,
            c_channels=D,
            n_blocks=n_blocks,
            n_layers=n_layers,
            skip_width=decoder_width,
            residual_width=2 * decoder_width,
        )
        self.codebook = VQEmbedding(K, D)

    def encode(self, m: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c_l = F.one_hot(y, self.n_classes).float().to(m.device)
        z = self.encoder(m, [c_l])
        return z

    def forward(self, m: torch.Tensor, y: torch.Tensor):
        z = self.encode(m, y)
        z_q_m_st, z_q_m = self.codebook.straight_through(z)
        m = shift1d(m, -1)
        S_tilde = self.decoder(m, z_q_m_st)
        return S_tilde, z, z_q_m

    def infer(self, m: torch.Tensor, y: torch.Tensor, destroy: float = 0):
        z = self.encode(m, y)
        if destroy > 0:
            z = destroy_along_channels(z, destroy)
        m = shift1d(m, -1)
        S_tilde = self.decoder(m, z)
        return S_tilde

    def test(self, x: torch.Tensor, S: torch.Tensor):
        β = 1.1
        m, y = x
        S_tilde, z_e_m, z_q_m = self(m, y)
        S = S[:, 0, :].to(S_tilde.device)

        # Reconstruction loss
        self.ℒ.recon = F.cross_entropy(S_tilde, S)

        # Vector quantization objective
        self.ℒ.vq = F.mse_loss(z_q_m, z_e_m.detach())

        # Commitment objective
        self.ℒ.commit = F.mse_loss(z_e_m, z_q_m.detach())

        ℒ = self.ℒ.recon + self.ℒ.vq + β * self.ℒ.commit
        return ℒ
