from math import tau as τ

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import sawtooth, square
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from ..data.toy import ToyData

PRINT_LENGTH = 500


def fig_summary(fname: str):
    df = pd.read_pickle(fname)
    df.destroy = df.destroy.astype("category")
    zer = df[df.destroy == 0.0].iloc[0].destroy

    fig = plt.figure(figsize=(5, 8))

    ax1 = fig.add_subplot(3, 2, 1)
    plt.title("Influence of the latent embedding")
    sns.scatterplot(
        x="periodicity",
        y="loss",
        hue="destroy",
        data=df,
        palette=["r", "g", "b"],
        ax=ax1,
    )

    ax2 = fig.add_subplot(3, 2, 3)
    plt.title("Mean loss over different destroy and source shapes")
    sns.boxplot(x="destroy", y="loss", hue="shape", data=df, ax=ax2)

    ax3 = fig.add_subplot(3, 2, 5)
    plt.title("Loss over shapes @ no destroy ")
    sns.scatterplot(
        x="periodicity", y="loss", hue="shape", data=df[df.destroy == zer],
        ax=ax3
    )

    ax4 = fig.add_subplot(2, 2, 2)
    plt.title("Mean period/loss for a sample @ no destroy")
    rdf = df[df.destroy == zer].groupby("n").agg("mean")
    sns.relplot(x="periodicity", y="loss", data=rdf, ax=ax4)

    ax5 = fig.add_subplot(2, 2, 4)
    plt.title("Sum period/loss for a sample @ no destroy")
    rdf = df[df.destroy == zer].groupby("n").agg("sum")
    sns.relplot(x="periodicity", y="loss", data=rdf, ax=ax5)

    return fig


def squeeze(tensor):
    tensor = tensor.detach().cpu().squeeze()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor[:, 100:PRINT_LENGTH].numpy()


def reconstruction(*signals: torch.Tensor):
    colores = ["k", "n", "y", "g", "r"]
    signals = list(map(squeeze, signals))
    ch = set(s.shape[0] for s in signals)
    N, hasM = max(ch), len(ch) >= 2
    ylim = (min(map(np.min, signals)), max(map(np.max, signals)))

    fig, axs = plt.subplots(N+hasM, sharex="all")
    for k, signal in enumerate(signals):
        if signal.shape[0] < N:
            axs[-1].plot(signal[0, :], c="k")
        else:
            c = colores[k % len(colores)]
            for i in range(N):
                axs[i].plot(signal[i, :], f"{c}-")
                axs[i].set_ylim(ylim)
    return fig


def prepare_plot_freq_loss(
        model: nn.Module,
        data: ToyData,
        ns: int,
        μ: int,
        destroy: float = 0.0,
        single: bool = False,
        device: str = "cpu",
) -> pd.DataFrame:
    d = {"n": [], "shape": [], "loss": [], "periodicity": [], "destroy": []}
    model = model.to(device)
    model.eval()
    for n in trange(len(data.data)):
        mix, stems = data[n]
        mix = mix.unsqueeze(0)
        prms = data.data[n]["params"]

        logits = model.infer(mix, stems)
        # TODO: FIX THIS
        logits = logits.cpu()
        for i in range(ns):
            d["n"].append(n)
            d["shape"].append(prms[i]["shape"])
            d["destroy"].append(destroy)
            d["periodicity"].append(prms[i]["φ"])
            d["loss"].append(
                F.cross_entropy(
                    logits[:, i * μ: (i + 1) * μ, :], stems[None, i, :]
                ).item()
            )
    df = pd.DataFrame(data=d)
    return df


def plot_freq_loss(fname: str):
    df = pd.read_pickle(fname)
    df.destroy = df.destroy.astype("category")
    # zer = df.destroy.iloc[0]
    sns.scatterplot(x="periodicity", y="loss", hue="shape", data=df)
    plt.show()


def add_plot_tick(ax, symbol, pos=0.5, where='tensor', size=0.05):

    if 'tensor' in where:
        anchor, loc = (pos, 1.01), 8
    else:
        anchor, loc = (-0.025, pos), 7

    _ax = inset_axes(ax, width=size, height=size, bbox_transform=ax.transAxes, bbox_to_anchor=anchor, loc=loc)
    _ax.axison = False

    x = np.linspace(0, τ)

    if 'sin' in symbol:
        _ax.plot(x, np.sin(x), c='k')
    elif 'tri' in symbol:
        _ax.plot(x, sawtooth(x, width=0.5), c='k')
    elif 'saw' in symbol:
        _ax.plot(x, sawtooth(x, width=1.), c='k')
    elif 'sq' in symbol:
        _ax.plot(x, square(x), c='k')
    else:
        raise ValueError("unknown symbol")


def plot_signal_heatmap(data, symbols):
    n = len(symbols)
    assert data.shape[0] == n == data.shape[1]

    fig, ax = plt.subplots()
    ax.axison = False
    ax.imshow(data, norm=colors.SymLogNorm(linthresh=0.03))

    pos_tick = np.linspace(0, 1, 2*n+1)[1::2]
    size = 1/n * 2.5

    for i in range(n):
        add_plot_tick(ax, symbols[i], pos=pos_tick[i], where='tensor', size=size)
        add_plot_tick(ax, symbols[i], pos=pos_tick[-i-1], where='y', size=size)
    return fig

