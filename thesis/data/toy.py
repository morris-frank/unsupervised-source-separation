from glob import glob
from random import randint
from typing import Dict
from typing import Union

import numpy as np
import torch

from ..audio import rand_period_phase, oscillator
from ..data import Dataset


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
        length: int = False,
    ):
        super(ToyData, self).__init__(n_mels=265)
        self.files = glob(f"{path}/{subset}/*npy")
        self.mix, self.mel_mix = mix, mel_mix
        self.rand_A = rand_amplitude
        self.noise = noise
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
            sources = (A + (1.0 - self.rand_A)) * sources
            mix = sources.mean(0, keepdim=True)

        if self.noise > 0:
            noise = self.noise * np.random.rand() * torch.randn_like(sources)
            sources = (sources + noise).clamp(-1, 1)
            mix = sources.mean(0, keepdim=True)

        sources = self._mel_get(sources, self.source, self.mel_source)
        mix = self._mel_get(mix, self.mix, self.mel_mix)

        if self.mix:
            if self.source:
                return mix, sources
            return mix
        else:
            return sources


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
        high, low = 65, 1
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
