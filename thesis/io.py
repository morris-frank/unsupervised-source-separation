import time
import os
import warnings
from collections import OrderedDict
from glob import glob
from os import path
from typing import Any, Type
from typing import Optional as Opt
from pathlib import Path

import torch
from random import random
from colorama import Fore
from torch import nn
from torch.serialization import SourceChangeWarning

from .setup import DEFAULT_CHECKPOINTS
from .utils import get_func_arguments

warnings.simplefilter("ignore", SourceChangeWarning)


def get_newest_file(folder: str, match: str = "*pt"):
    chosen = sorted(glob(f"{folder}/{match}"), key=lambda x: path.getmtime(x))[-1]
    print(
        f"{Fore.YELLOW}For {Fore.GREEN}{match} {Fore.YELLOW}we using\t{Fore.GREEN}{chosen}{Fore.RESET}"
    )
    return chosen


def get_newest_checkpoint(match: str):
    if "*" not in match:
        return match
    if not match.endswith("pt"):
        if not match.endswith("*"):
            match += '*'
        match += "pt"
    return get_newest_file(DEFAULT_CHECKPOINTS, match)


def glob_remove(path: str):
    for fp in glob(path):
        os.remove(fp)


def save_append(fp: str, obj: Any):
    """
    Appends to a pickled torch save. Create file if not exists.
    Args:
        fp: Path to file
        obj: New obj to append
    """
    fp = path.abspath(fp)
    if path.exists(fp):
        data = torch.load(fp)
    else:
        data = [obj]
    data.append(obj)
    torch.save(data, fp)


def load_model(fp: str, device: str, train: bool = False, model_class: Opt[Type] = None ) -> nn.Module:
    save_point = torch.load(fp, map_location=torch.device(device))

    if "model_state_dict" in save_point:
        state_dict = save_point["model_state_dict"]

        if model_class is None:
            model_class = save_point["params"]["__class__"]
        args = save_point["params"]["args"]
        kwargs = save_point["params"]["kwargs"].copy()
        model = model_class(*args, **kwargs)

        for k in filter(lambda x: x.startswith("p_s."), list(state_dict.keys())):
            del state_dict[k]

        if next(iter(state_dict.keys())).startswith("module."):
            _state_dict = OrderedDict({k[7:]: v for k, v in state_dict.items()})
            state_dict = _state_dict

        model.load_state_dict(state_dict)
    else:
        model = save_point["model"]

    if not train:
        model.eval()
        return model.to(device)

    # TODO implement continue training load
    raise NotImplementedError


class FileLock(object):
    def __init__(self, file_name):
        self.path = Path(path.normpath(file_name) + '.lock')

    def __enter__(self):
        while self.path.exists():
            print(f'{Fore.YELLOW}LockFile exists. Waiting…')
            time.sleep(random())
        time.sleep(random())
        self.path.touch()

    def __exit__(self, *args):
        if self.path.exists():
            self.path.unlink()
        else:
            print(f"{Fore.RED}Trying to delete not-existing FileLock{Fore.RESET}")


def exit_prompt():
    inp = input("REPL? \t")
    if inp.lower().strip() == "q":
        exit()
    elif inp.lower().strip() == "h":
        import ipdb

        ipdb.set_trace()


def vprint(*args):
    names = get_func_arguments()
    for n,v in zip(names, args):
        if isinstance(v, torch.Tensor):
            v = v.shape
        print(f'{Fore.YELLOW}{n}{Fore.WHITE} = {Fore.MAGENTA}{v}', end=f'{Fore.RESET}\t')
    print()

