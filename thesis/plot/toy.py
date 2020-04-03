from math import tau as τ

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import sawtooth, square

from ..utils import get_func_arguments

PRINT_START = 100
PRINT_LENGTH = 500


def squeeze(tensor):
    tensor = tensor.detach().cpu().squeeze()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor[:, PRINT_START : PRINT_START + PRINT_LENGTH].numpy()


def reconstruction(*signals: torch.Tensor, sharey: bool = True, ylim=None):
    arguments = get_func_arguments()
    colores = ["k", "n", "y", "g", "r"]
    signals = list(map(squeeze, signals))
    ch = set(s.shape[0] for s in signals)
    N, hasM = max(ch), len(ch) >= 2
    if not ylim:
        ylim = (min(map(np.min, signals)), max(map(np.max, signals)))

    fig, axs = plt.subplots(N + hasM, sharex="all", sharey="none", squeeze=False)
    if not isinstance(axs, np.ndarray):
        axs = list(axs)
    for k, (signal, name) in enumerate(zip(signals, arguments)):
        if signal.shape[0] < N:
            axs[-1, 0].plot(signal[0, :], c="k", label=name)
        else:
            c = colores[k % len(colores)]
            for i in range(N):
                axs[i, 0].plot(signal[i, :], f"{c}-", label=name)
                if sharey:
                    axs[i, 0].set_ylim(ylim)
                # else:
                #     axs[i, 0].relim()
                #     axs[i, 0].autoscale_view()
    for ax in axs.flatten().tolist():
        ax.legend()
    return fig


def add_plot_tick(ax, symbol, pos=0.5, where="tensor", size=0.05):

    if "tensor" in where:
        anchor, loc = (pos, 1.01), 8
    else:
        anchor, loc = (-0.025, pos), 7

    _ax = inset_axes(
        ax,
        width=size,
        height=size,
        bbox_transform=ax.transAxes,
        bbox_to_anchor=anchor,
        loc=loc,
    )
    _ax.axison = False

    x = np.linspace(0, τ)

    if "sin" in symbol:
        _ax.plot(x, np.sin(x), c="k")
    elif "tri" in symbol:
        _ax.plot(x, sawtooth(x, width=0.5), c="k")
    elif "saw" in symbol:
        _ax.plot(x, sawtooth(x, width=1.0), c="k")
    elif "sq" in symbol:
        _ax.plot(x, square(x), c="k")
    else:
        raise ValueError("unknown symbol")


def plot_signal_heatmap(data, symbols):
    n = len(symbols)
    assert data.shape[0] == n == data.shape[1]

    fig, ax = plt.subplots()
    ax.axison = False
    ax.imshow(data, norm=colors.SymLogNorm(linthresh=0.03))

    pos_tick = np.linspace(0, 1, 2 * n + 1)[1::2]
    size = 1 / n * 2.5

    for i in range(n):
        add_plot_tick(ax, symbols[i], pos=pos_tick[i], where="tensor", size=size)
        add_plot_tick(ax, symbols[i], pos=pos_tick[-i - 1], where="y", size=size)
    return fig
