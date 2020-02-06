from math import pow
from os.path import abspath

from torch import nn

from . import Dataset
from .toy import ToyDataSingle, ToyData


def map_dataset(model: nn.Module, data_dir: abspath, subset: str) -> Dataset:
    if model.__class__.__name__ == 'MultiRealNVP':
        wn_layers = model.params['kwargs']['wn_layers']
        receptive_field = int(2 * pow(2, wn_layers - 1))
        dset = ToyData(f'{data_dir}/toy_{subset}.npy', receptive_field)
    elif model.__class__.__name__ in ('ConditionalRealNVP',
                                      'ExperimentalRealNVP'):
        wn_layers = model.params['kwargs']['wn_layers']
        receptive_field = int(2 * pow(2, wn_layers - 1))
        dset = ToyDataSingle(f'{data_dir}/toy_{subset}.npy', receptive_field)
    elif model.__class__.__name__ in ('WaveGlow',
                                      'ExperimentalWaveGlow'):
        wn_layers = model.params['kwargs']['wn_layers']
        receptive_field = int(pow(2, wn_layers - 1))
        dset = ToyData(f'{data_dir}/toy_{subset}.npy', receptive_field)
    else:
        raise ValueError('Unrecognized Model Class – do it on your own pls')
    return dset
