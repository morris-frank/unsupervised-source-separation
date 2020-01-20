from typing import Dict, Any


def clean_init_args(_locals: Dict) -> Dict[str, Any]:
    args = dict(args=(), kwargs={}, __class__=None)
    del _locals['self']
    args['__class__'] = _locals.pop('__class__')
    args['args'] = _locals.pop('args')
    args['kwargs'] = _locals.pop('kwargs')
    args['kwargs'] = {**args['kwargs'], **_locals}
    return args
