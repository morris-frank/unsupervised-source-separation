from glob import glob
from typing import Dict
from typing import Union

import numpy as np
import torch

from ..audio import rand_period_phase, oscillator
from ..data import Dataset

from random import uniform, randint


class ToyData(Dataset):
    def __init__(
        self,
        path: str,
        subset: str,
        mix: bool = False,
        mel_mix: bool = False,
        source: Union[bool, int] = False,
        mel_source: Union[bool, int] = False,
        rand_amplitude: float = 0.0,
        noise: float = 0.0,
        with_phase: int = False,
        length: int = False,
        **kwargs,
    ):
        super(ToyData, self).__init__(n_mels=79 if with_phase else 80, **kwargs)
        self.files = glob(f"{path}/{subset}/*npy")
        self.mix, self.mel_mix = mix, mel_mix
        self.rand_A = rand_amplitude
        self.noise = noise
        self.with_phase = with_phase
        self.length = length

        self.source, self.mel_source = source is not False, mel_source is not False
        if self.source is True and self.mel_source is True:
            assert source == mel_source
        if self.source is True:
            self.k = "all" if isinstance(source, bool) else source
        else:
            self.k = "all" if isinstance(mel_source, bool) else mel_source

        assert mix or mel_mix or self.source or self.mel_source

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        datum = np.load(self.files[idx], allow_pickle=True).item()
        mix = torch.tensor(datum["mix"], dtype=torch.float32).unsqueeze(0)
        sources = torch.tensor(datum["sources"], dtype=torch.float32)

        if self.k != "all":
            sources = sources[None, self.k, :].contiguous()

        if self.length is not False:
            L = mix.shape[-1]
            ν = randint(0, L - self.length)
            mix = mix[..., ν : ν + self.length]
            sources = sources[..., ν : ν + self.length]

        if self.rand_A > 0:
            A = torch.rand(sources.shape[0], 1) * self.rand_A
            sources = (A + (1. - self.rand_A)) * sources
            mix = sources.mean(0, keepdim=True)

        if self.noise > 0:
            noise = self.noise * torch.randn_like(sources)
            sources = (sources + noise).clamp(-1, 1)
            mix = sources.mean(0, keepdim=True)

        sources = self._mel_get(sources, self.source, self.mel_source)
        mix = self._mel_get(mix, self.mix, self.mel_mix)

        if self.mel_source and self.with_phase:
            add = torch.ones(4, 1, sources.shape[-1]) * torch.tensor(datum["φ"]).view(
                4, 1, 1
            )
            sources = (sources[0], torch.cat([sources[1], add], dim=1))

        if self.mix:
            if self.source:
                return mix, sources
            return mix
        else:
            return sources


class ToyDataAndNoise(ToyData):
    def __getitem__(self, idx):
        if idx % 20 == 0:
            σ = uniform(0, 0.1)
            s = σ * torch.randn(1, 3072)
            m = self.spectrograph(s.squeeze())
            return (s, m), -0.01
        else:
            return super(ToyDataAndNoise, self).__getitem__(idx), 1


class RandToyData(ToyData):
    def __getitem__(self, idx):
        if idx % 20 == 0:
            σ = uniform(0, 0.1)
            s = σ * torch.randn(1, 3072)
            m = self.spectrograph(s.squeeze())
            return (s, m), -1
        elif idx % 21 == 0:
            s, m = super(RandToyData, self).__getitem__(idx)
            s = s.mean(0, keepdim=True)
            m = self.spectrograph(s.squeeze())
            return (s, m), -1
        else:
            s, m = super(RandToyData, self).__getitem__(idx)
            k = randint(0, 3)
            s = s[None, k, :]
            m = m[k, :]
            return (s, m), k


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
