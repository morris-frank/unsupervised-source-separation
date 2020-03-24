import warnings
from collections import OrderedDict
from os.path import abspath, exists
from typing import Any

import librosa
import torch
from torch import nn
from torch.serialization import SourceChangeWarning

from .functional import encode_μ_law

warnings.simplefilter("ignore", SourceChangeWarning)


def save_append(fp: str, obj: Any):
    """
    Appends to a pickled torch save. Create file if not exists.
    Args:
        fp: Path to file
        obj: New obj to append
    """
    fp = abspath(fp)
    if exists(fp):
        data = torch.load(fp)
    else:
        data = [obj]
    data.append(obj)
    torch.save(data, fp)


def load_audio(fp: str) -> torch.Tensor:
    """
    Loads an audio file with librosa and output μ-law encoded.

    Args:
        fp: Path to file

    Returns:
        Content in μ-law
    """
    raw, sr = librosa.load(fp, mono=True, sr=None)
    assert sr == 16000
    raw = torch.tensor(raw[None, None, ...], dtype=torch.float32)
    x = encode_μ_law(raw) / 128
    return x


def load_model(fp: str, device: str, train: bool = False) -> nn.Module:
    """

    Args:
        fp:
        device:
        train:

    Returns:

    """
    save_point = torch.load(fp, map_location=torch.device(device))

    if "model_state_dict" in save_point:
        state_dict = save_point["model_state_dict"]

        model_class = save_point["params"]["__class__"]
        args = save_point["params"]["args"]
        kwargs = save_point["params"]["kwargs"].copy()
        model = model_class(*args, **kwargs)

        if next(iter(state_dict.keys())).startswith("module."):
            _state_dict = OrderedDict({k[7:]: v for k, v in state_dict.items()})
            state_dict = _state_dict

        model.load_state_dict(state_dict)
    else:
        model = save_point["model"]

    if not train:
        model.eval()
        return model

    # TODO implement continue training load
    raise NotImplementedError
