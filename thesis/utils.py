import inspect
import re
from collections import defaultdict
from itertools import product
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


class _LossLogger(object):
    def __init__(self):
        self.log = defaultdict(list)

    def __setattr__(self, key: str, value: Any):
        super(_LossLogger, self).__setattr__(key, value)
        if key != "log":
            if isinstance(value, torch.Tensor):
                value = value.detach().mean().item()
            self.log[key].append(value)

    def clear(self):
        attrs = set(self.__dict__.keys()) - {"log"}
        for attr in attrs:
            del self.__dict__[attr]


def get_func_arguments():
    func_name = inspect.stack()[1].function.strip()
    code_line = inspect.stack()[2].code_context[0].strip()
    try:
        argument_string = re.search(rf"{func_name}\((.*)\)", code_line)[1]
    except TypeError:
        import ipdb

        ipdb.set_trace()
    arguments = re.split(r",\s*(?![^()]*\))", argument_string)
    return arguments


def any_invalid_grad(parameters):
    parameters = list(filter(lambda x: x.grad is not None, parameters))
    any_is_invalid = any(
        torch.isnan(p.grad.data).any() or torch.isinf(p.grad.data).any()
        for p in parameters
    )
    return any_is_invalid


def max_grad(parameters):
    parameters = list(filter(lambda x: x.grad is not None, parameters))
    max_value = max(p.grad.data.abs().max() for p in parameters)
    return max_value
