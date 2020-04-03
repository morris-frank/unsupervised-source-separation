from math import log
from math import pi as π
from typing import Tuple

import numpy as np
from numpy.random import randint
import torch
from scipy import signal


def oscillator(length: int, shape: str, ν: int, φ: int = 0) -> np.ndarray:
    """
    :param length: The length of the signal
    :param shape: Which curve to use: 'tri', 'saw', 'sq', 'sine'
    :param ν: The length of one period
    :param φ: The phase in [0, ν]
    :return:
    """
    assert 0 <= φ <= ν
    shape = shape.lower()
    x = np.linspace(0, length + ν - 1, length + ν)
    if shape == "triangle":
        y = signal.sawtooth(2 * π * x / ν, width=0.5)
    elif shape == "saw":
        y = signal.sawtooth(2 * π * x / ν, width=1.0)
    elif shape == "reversesaw":
        y = signal.sawtooth(2 * π * x / ν, width=0.0)
    elif shape == "square":
        y = signal.square(2 * π * x / ν)
    elif shape == "halfsin":
        _y = np.zeros(ν)
        _y[: ν // 2] = np.sin(2 * π * np.linspace(0, 1, ν // 2))
        y = np.tile(_y, x.shape[0] // ν + 1)
    elif shape == "noise":
        y = np.random.rand(*x.shape)
        y *= 0.1
    elif shape == "sin":
        y = np.sin(2 * π * x / ν)
    else:
        raise ValueError("Invalid shape given")
    y = y[φ : φ + length][None, ...]
    return y


def key2freq(n: int) -> float:
    """
    Gives the frequency for a given piano key.
    Args:
        n: The piano key index

    Returns:
        The Frequency
    """
    return 440 * 2 ** ((n - 49) / 12)


def rand_period_phase(high: int = 88, low: int = 1, sr: int = 16000) -> Tuple[int, int]:
    key = randint(low, high * 10, 1)[0] / 10
    freq = key2freq(key)
    ν = int(sr // freq)
    φ = randint(0, ν, 1)[0]
    return ν, φ


def mel_spectrogram(waveform):
    from .nn.modules import MelSpectrogram

    return MelSpectrogram()(waveform)


def encode_μ_law(waveform: torch.Tensor, μ: int = 255) -> torch.Tensor:
    """
    Encodes the input tensor element-wise with μ-law encoding

    Args:
        waveform: tensor
        μ: the size of the encoding (number of possible classes)

    Returns:
        the encoded tensor
    """
    assert μ & 1
    assert waveform.max() <= 1.0 and waveform.min() >= -1.0
    μ -= 1
    hμ = μ // 2
    out = torch.sign(waveform) * torch.log(1 + μ * torch.abs(waveform)) / log(μ)
    out = torch.round(out * hμ) + hμ
    return out


def decode_μ_law(waveform: torch.Tensor, μ: int = 255) -> torch.Tensor:
    """
    Applies the element-wise inverse μ-law encoding to the tensor.

    Args:
        waveform: input tensor
        μ: size of the encoding (number of possible classes)

    Returns:
        the decoded tensor
    """
    assert μ & 1
    μ = μ - 1
    hμ = μ // 2
    out = (waveform.type(torch.float32) - hμ) / hμ
    out = torch.sign(out) / μ * (torch.pow(μ, torch.abs(out)) - 1)
    return out
