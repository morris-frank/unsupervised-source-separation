import random
from glob import glob
from typing import Optional

import numpy as np
import torch

from ..data import Dataset
from ..functional import encode_μ_law


def μ_enc(
    mix: np.ndarray, sources: np.ndarray, μ: Optional[int] = None,
):
    assert μ & 1
    hμ = (μ - 1) // 2
    mix = encode_μ_law(mix, μ=μ) / hμ
    sources = encode_μ_law(sources, μ=μ)
    sources = (sources + hμ).long()
    return mix, sources


class ToyData(Dataset):
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

    def __getitem__(self, idx: int):
        datum = self.get(idx)
        return datum["mix"], datum["sources"]


class ToyDataSourceK(ToyData):
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


class ToyDataSources(ToyData):
    def __init__(self, mel: bool = False, *args, **kwargs):
        super(ToyDataSources, self).__init__(*args, **kwargs)
        self.mel = mel

    def __getitem__(self, idx: int):
        datum = self.get(idx)
        sources = datum["sources"].unsqueeze(1)  # Add an empty channel dim

        if self.mel:
            mel = datum["mel_sources"]
            return sources, mel
        else:
            return sources


class ToyDataMixes(ToyData):
    def __init__(self, mel: bool = False, sources: bool = False, *args, **kwargs):
        super(ToyDataMixes, self).__init__(*args, **kwargs)
        self.mel = mel
        self.sources = sources

    def __getitem__(self, idx: int):
        datum = self.get(idx)
        mix = datum["mix"].contiguous()
        mel = datum["mel_mix"].contiguous()
        sources = datum["sources"].contiguous()

        if self.sources:
            if self.mel:
                return (mix, mel), sources
            else:
                return mix, sources
        else:
            if self.mel:
                return mix, mel
            else:
                return mix
