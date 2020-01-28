from os.path import abspath, exists
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
    del _locals['self']
    args['__class__'] = _locals.pop('__class__')
    if 'args' in _locals:
        args['args'] = _locals.pop('args')
    if 'kwargs' in _locals:
        args['kwargs'] = _locals.pop('kwargs')
    args['kwargs'] = {**args['kwargs'], **_locals}
    return args


def save_append(fname: str, obj: Any):
    """
    Appends to a pickled torch save. Create file if not exists.
    Args:
        fname: Path to file
        obj: New obj to append
    """
    fp = abspath(fname)
    if exists(fp):
        data = torch.load(fp)
    else:
        data = [obj]
    data.append(obj)
    torch.save(data, fp)
