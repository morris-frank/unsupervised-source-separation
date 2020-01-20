from collections import OrderedDict
from typing import Tuple

import librosa
import torch
from torch import nn

from .functional import encode_μ_law
from .modules import AutoEncoder


def load_model(fp: str, device: str, model: nn.Module, train: bool = False) \
        -> nn.Module:
    """

    Args:
        fp:
        device:
        model:
        train:

    Returns:

    """
    # TODO: DEPRECATE THE model arg
    save_point = torch.load(fp, map_location=torch.device(device))
    state_dict = save_point['model_state_dict']

    if 'params' in save_point:
        _class = save_point['params']['__class__']
        args = save_point['params']['args'].copy()
        kwargs = save_point['params']['kwargs'].copy()
        model = _class(*args, **kwargs)

    if next(iter(state_dict.keys())).startswith('module.'):
        _state_dict = OrderedDict({k[7:]: v for k, v in state_dict.items()})
        state_dict = _state_dict

    model.load_state_dict(state_dict)

    if not train:
        return model

    raise NotImplementedError


def load_audio(fp: str) -> torch.Tensor:
    """

    Args:
        fp:

    Returns:

    """
    raw, sr = librosa.load(fp, mono=True, sr=None)
    assert sr == 16000
    raw = torch.tensor(raw[None, None, ...], dtype=torch.float32)
    x = encode_μ_law(raw) / 128
    return x


def generate(model: AutoEncoder, x: torch.Tensor, length: int, device: str) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """

    Args:
        model:
        x:
        length:
        device:

    Returns:

    """
    model.eval()
    decoder = model.decoder
    embedding = model.encoder(x).mean(-1).unsqueeze(-1)

    # Build and upsample all the conditionals from the embedding:
    l_conds = [l_cond(embedding) for l_cond in decoder.conds]
    l_conds.append(decoder.final_cond(embedding))

    generation = decoder.generate(x, l_conds, length, device, 1)

    return generation.cpu(), embedding.cpu()
