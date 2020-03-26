import random
from glob import glob
from typing import Optional

import numpy as np
import torch

from ..data import Dataset


TOY_SIGNALS = ["sin", "square", "saw", "triangle"]


class _ToyData(Dataset):
    def __init__(self, path: str, crop: Optional[int] = None):
        self.files = glob(f"{path}/*npy")
        self.crop = crop

    def __len__(self):
        return len(self.files)

    def get(self, idx: int):
        datum = np.load(self.files[idx], allow_pickle=True).item()
        if self.crop:
            mix_w = datum["mix"].size
            p = random.randint(0, mix_w - self.crop)
            datum["mix"] = datum["mix"][p : p + self.crop]
            datum["sources"] = datum["sources"][:, p : p + self.crop]
            mel_w = datum["mel_mix"].shape[0]
            l_m, r_m = int(p / mix_w * mel_w), int((p + self.crop) / mix_w * mel_w)
            datum["mel_mix"] = datum["mel_mix"][l_m : r_m + 1, :]
            datum["mel_sources"] = datum["mel_sources"][:, l_m : r_m + 1, :]
        datum["mix"] = torch.tensor(datum["mix"], dtype=torch.float32).unsqueeze(0)
        datum["sources"] = torch.tensor(datum["sources"], dtype=torch.float32)
        datum["mel_mix"] = torch.tensor(
            datum["mel_mix"], dtype=torch.float32
        ).transpose(0, 1)
        datum["mel_sources"] = torch.tensor(
            datum["mel_sources"], dtype=torch.float32
        ).transpose(1, 2)
        return datum


class ToyData(_ToyData):
    def __init__(
        self,
        mix: bool = True,
        mel: bool = False,
        sources: bool = False,
        mel_sources: bool = False,
        *args,
        **kwargs,
    ):
        super(ToyData, self).__init__(*args, **kwargs)
        assert mix or sources
        assert mix or not mel
        assert sources or not mel_sources
        self.mix, self.mel = mix, mel
        self.sources, self.mel_sources = sources, mel_sources

    @staticmethod
    def _mel_get(datum, name, mel):
        if mel:
            return datum[name].contiguous(), datum[f"mel_{name}"].contiguous()
        else:
            return datum[name].contiguous()

    def __getitem__(self, idx: int):
        datum = self.get(idx)

        if self.sources:
            sources = self._mel_get(datum, "sources", self.mel_sources)

        if self.mix:
            mix = self._mel_get(datum, "mix", self.mel)
            if self.sources:
                return mix, sources
            return mix
        else:
            return sources


class ToyDataRandomAmplitude(ToyData):
    def __init__(self, min: float = .25, *args, **kwargs):
        super(ToyDataRandomAmplitude, self).__init__(mix=True, mel=True, sources=True, *args, **kwargs)
        self.min = min

    def __getitem__(self, idx):
        (_, m_mel), s = super(ToyDataRandomAmplitude, self).__getitem__(idx)
        # Generate some random amplitudes
        A = torch.rand(s.shape[0], 1) * (1. - self.min) + self.min
        s = A * s
        m = s.mean(0, keepdim=True)
        return (m, m_mel), s


class ToyDataSourceK(_ToyData):
    def __init__(self, k: int, mel: bool = False, *args, **kwargs):
        super(ToyDataSourceK, self).__init__(*args, **kwargs)
        self.k, self.mel = k, mel

    def __getitem__(self, idx: int):
        datum = self.get(idx)
        source = datum["sources"][None, self.k, :].contiguous()

        if self.mel:
            mel = datum["mel_sources"][self.k, ...].contiguous()
            return source, mel
        else:
            return source
