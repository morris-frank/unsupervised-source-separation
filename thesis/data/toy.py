from glob import glob
from typing import Dict
from typing import Union

import numpy as np
import torch

from ..audio import rand_period_phase, oscillator
from ..data import Dataset
from ..nn.modules import MelSpectrogram

from random import uniform


class ToyData(Dataset):
    def __init__(
        self,
        path: str,
        mix: bool = False,
        mel: bool = False,
        source: Union[bool, int] = False,
        mel_source: bool = False,
        rand_amplitude: float = 0.0,
        noise: float = 0.0,
        rand_noise: bool = False,
    ):
        super(ToyData, self).__init__()
        self.files = glob(f"{path}/*npy")
        self.mix, self.mel = mix, mel
        self.rand_amplitude = rand_amplitude
        self.noise, self.rand_noise = noise, rand_noise

        self.k = "all" if isinstance(source, bool) else source
        self.source = source is not False
        self.mel_source = mel_source

        if mel_source or mel:
            self.melspec = MelSpectrogram()

        assert mix or self.source
        assert mix or not mel
        assert self.source or not mel_source

    def __len__(self):
        return len(self.files)

    def _mel_get(self, tensor, compute_mel):
        if compute_mel:
            mel = self.melspec(tensor.squeeze())
            return tensor, mel
        else:
            return tensor

    def __getitem__(self, idx: int):
        datum = np.load(self.files[idx], allow_pickle=True).item()
        mix = torch.tensor(datum["mix"], dtype=torch.float32).unsqueeze(0)
        sources = torch.tensor(datum["sources"], dtype=torch.float32)

        if self.k != "all":
            sources = sources[None, self.k, :].contiguous()

        if self.rand_amplitude > 0:
            A = torch.rand(sources.shape[0], 1) * self.rand_amplitude + (
                1.0 - self.rand_amplitude
            )
            sources = A * sources
            mix = sources.mean(0, keepdim=True)

        if self.noise > 0:
            σ = uniform(0, self.noise) if self.rand_noise else self.noise
            noise = σ * torch.randn_like(sources)
            sources = (sources + noise).clamp(-1, 1)
            mix = sources.mean(0, keepdim=True)

        mix = self._mel_get(mix, self.mel)
        sources = self._mel_get(sources, self.mel_source)

        if self.mix:
            if self.source:
                return mix, sources
            return mix
        else:
            return sources


class ToyDataAndNoise(ToyData):
    def __getitem__(self, idx):
        if idx % 20 == 0:
            print('noise')
            σ = uniform(0, 0.1)
            s = σ * torch.randn(1, 3072)
            m = self.melspec(s.squeeze())
            return (s, m), -1
        else:
            return super(ToyDataAndNoise, self).__getitem__(idx), 1


def generate_toy(length: int, ns: int) -> Dict:
    signals = []
    shapes = [
        "sin",
        "square",
        "saw",
        "triangle",
        "halfsin",
        "low_square",
        "reversesaw",
        "noise",
    ]
    item = {"shape": [], "ν": [], "φ": []}
    if ns > 5:
        shapes[1] = "high_square"
    for i, s in zip(range(ns), shapes):
        item["shape"].append(s)
        high, low = 88, 1
        if s.startswith("high_"):
            low = 50
            s = s[5:]
        if s.startswith("low_"):
            high = 40
            s = s[4:]
        ν, φ = rand_period_phase(high, low)
        signals.append(oscillator(length, s, ν, φ))
        item["ν"].append(ν)
        item["φ"].append(φ)
    sources = np.concatenate(signals, axis=0)
    λ = np.ones(ns) / ns
    mix = λ @ sources

    return {"sources": sources, "mix": mix, **item}
