from math import pow
from os.path import abspath

from torch import nn

from . import Dataset
from .toy import ToyData, ToyDataSpectral, ToyDataSingleSourceOnly


def map_dataset(model: nn.Module, data_dir: abspath, subset: str) -> Dataset:
    if model.__class__.__name__ in "RealNVP":
        wn_layers = model.params["kwargs"]["wn_layers"]
        receptive_field = int(2 * pow(2, wn_layers - 1))
        dset = ToyData(f"{data_dir}/toy_{subset}.npy", receptive_field)
    elif model.__class__.__name__ == "PriorNVP":
        wn_layers = model.params["kwargs"]["wn_layers"]
        receptive_field = int(2 * pow(2, wn_layers - 1))
        dset = ToyDataSingleSourceOnly(
            k=model.k, filepath=f"{data_dir}/toy_{subset}.npy", crop=receptive_field
        )
    elif model.__class__.__name__ == "WaveGlow":
        wn_layers = model.params["kwargs"]["wn_layers"]
        receptive_field = int(pow(2, wn_layers - 1))
        dset = ToyData(f"{data_dir}/toy_{subset}.npy", receptive_field)
    elif model.__class__.__name__ == "Hydra":
        receptive_field = 3 * 2 ** 10
        μ = model.out_channels
        dset = ToyData(f"{data_dir}/toy_{subset}.npy", receptive_field, μ=μ)
    elif model.__class__.__name__ == "MONet":
        receptive_field = 3 * 2 ** 10
        dset = ToyDataSpectral(f"{data_dir}/toy_{subset}.npy", receptive_field)
    else:
        raise ValueError("Unrecognized Model Class – do it on your own pls")
    return dset
