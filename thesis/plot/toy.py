from typing import Generator

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from ..data import Dataset
from ..data.toy import ToyData


def fig_summary(fname: str):
    df = pd.read_pickle(fname)
    df.destroy = df.destroy.astype('category')
    zer = df[df.destroy == 0.0].iloc[0].destroy

    fig = plt.figure(figsize=(5, 8))

    ax1 = fig.add_subplot(3, 2, 1)
    plt.title('Influence of the latent embedding')
    sns.scatterplot(x='periodicity', y='loss', hue='destroy', data=df,
                    palette=['r', 'g', 'b'], ax=ax1)

    ax2 = fig.add_subplot(3, 2, 3)
    plt.title('Mean loss over different destroy and source shapes')
    sns.boxplot(x="destroy", y="loss", hue="shape", data=df, ax=ax2)

    ax3 = fig.add_subplot(3, 2, 5)
    plt.title('Loss over shapes @ no destroy ')
    sns.scatterplot(x='periodicity', y='loss', hue='shape',
                    data=df[df.destroy == zer], ax=ax3)

    ax4 = fig.add_subplot(2, 2, 2)
    plt.title('Mean period/loss for a sample @ no destroy')
    rdf = df[df.destroy == zer].groupby('n').agg('mean')
    sns.relplot(x="periodicity", y="loss", data=rdf, ax=ax4)

    ax5 = fig.add_subplot(2, 2, 4)
    plt.title('Sum period/loss for a sample @ no destroy')
    rdf = df[df.destroy == zer].groupby('n').agg('sum')
    sns.relplot(x="periodicity", y="loss", data=rdf, ax=ax5)

    return fig


def plot_reconstruction(m: torch.Tensor, S: torch.Tensor,
                        S_tilde: torch.Tensor) -> plt.Figure:
    if S_tilde.ndim == 1:
        S_tilde = S_tilde.unsqueeze(0)
    p_length = 500
    n = S.shape[0]
    fig, axs = plt.subplots(n + 1, 2, sharex='all')
    for i in range(n):
        axs[i, 0].plot(S[i, 100:p_length], c='r')
        axs[i, 1].plot(S_tilde[i, 100:p_length], c='b')
        axs[-1, 0].plot(m[0, 100:p_length], c='g')
    return fig


def _tuple_unsequeeze(x):
    if isinstance(x, tuple):
        m = x[0]
        x = (x[0].unsqueeze(0), torch.tensor([x[1]]))
    else:
        m = x
        x = x.unsqueeze(0)
    return m, x


def example_reconstruction(model: nn.Module, data: Dataset) \
        -> Generator[plt.Figure, None, None]:
    for i, (x, S) in enumerate(data):
        m, x = _tuple_unsequeeze(x)
        S_tilde = model.infer(x).squeeze()
        fig = plot_reconstruction(m, S, S_tilde)
        fig.savefig(f'./figures/{type(model).__name__}_{i}.png')
        yield fig


def z_example_reconstruction(model: nn.Module, data: Dataset) \
        -> Generator[plt.Figure, None, None]:
    for i, (x, S) in enumerate(data):
        m, x = _tuple_unsequeeze(x)
        r = model(x, S.unsqueeze(0))
        z = r[0]  # account for multiple outputs
        S_tilde = model.infer(x, z=z).squeeze()
        fig = plot_reconstruction(m, S, S_tilde)
        fig.savefig(f'./figures/{type(model).__name__}_z_{i}.png')
        yield fig


def prepare_plot_freq_loss(model: nn.Module, data: ToyData, ns: int,
                           μ: int, destroy: float = 0.,
                           single: bool = False,
                           device: str = 'cpu') -> pd.DataFrame:
    d = {'n': [], 'shape': [], 'loss': [], 'periodicity': [], 'destroy': []}
    model = model.to(device)
    model.eval()
    for n in trange(len(data.data)):
        mix, stems = data[n]
        mix = mix.unsqueeze(0)
        prms = data.data[n]['params']

        logits = model.infer(mix, stems)
        # TODO: FIX THIS
        logits = logits.cpu()
        for i in range(ns):
            d['n'].append(n)
            d['shape'].append(prms[i]['shape'])
            d['destroy'].append(destroy)
            d['periodicity'].append(prms[i]['φ'])
            d['loss'].append(F.cross_entropy(logits[:, i * μ:(i + 1) * μ, :],
                                             stems[None, i, :]).item())
    df = pd.DataFrame(data=d)
    return df


def plot_freq_loss(fname: str):
    df = pd.read_pickle(fname)
    df.destroy = df.destroy.astype('category')
    # zer = df.destroy.iloc[0]
    sns.scatterplot(x='periodicity', y='loss', hue='shape', data=df)
    plt.show()
