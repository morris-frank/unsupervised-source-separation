from collections import defaultdict
from contextlib import contextmanager
from glob import glob
from itertools import product
from os.path import getmtime
import os
from typing import Dict, Any

import torch


def clean_init_args(_locals: Dict) -> Dict[str, Any]:
    """
    Prepares the local variables from call locals() to the fixed pickle form.
    To be used in ALL constructors of all model definitions.
    Args:
        _locals: call locals().copy()!

    Returns:
        dictionary of prepared and organized arguments

    """
    args = dict(args=(), kwargs={}, __class__=None)
    del _locals["self"]
    args["__class__"] = _locals.pop("__class__")
    if "args" in _locals:
        args["args"] = _locals.pop("args")
    if "kwargs" in _locals:
        args["kwargs"] = _locals.pop("kwargs")
    args["kwargs"] = {**args["kwargs"], **_locals}
    return args


def range_product(*args: int) -> product:
    """
    Gives an iterator over the product of the ranges of the given integers.

    Args:
        *args: A number of Integers

    Returns:
        the product iterator
    """
    return product(*map(range, args))


@contextmanager
def optional(condition, context_manager):
    if condition:
        with context_manager:
            yield
    else:
        yield


class _LossLogger(object):
    def __init__(self):
        self.log = defaultdict(list)

    def __setattr__(self, key: str, value: Any):
        super(_LossLogger, self).__setattr__(key, value)
        if key != "log":
            if isinstance(value, torch.Tensor):
                value = value.detach().mean().item()
            self.log[key].append(value)


def get_newest_file(folder: str):
    return sorted(glob(f"{folder}/*pt"), key=lambda x: getmtime(x))[-1]


def remove_glob(path: str):
    for fp in glob(path):
        os.remove(fp)
