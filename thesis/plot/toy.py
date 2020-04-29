from itertools import product
from math import tau as τ

from matplotlib import rcParams
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
    while tensor.dim() < 3:
        tensor = tensor.unsqueeze(0)
    return tensor[:, :, PRINT_START : PRINT_START + PRINT_LENGTH].numpy()


def reconstruction(*signals: torch.Tensor, sharey: bool = True, ylim=None):
    arguments = get_func_arguments()
    colores = ["k", "n", "y", "g", "r"]
    signals = list(map(squeeze, signals))
    ch = set(s.shape[1] for s in signals)
    C, hasM = max(ch), len(ch) >= 2
    N = max(s.shape[0] for s in signals)
    if not ylim:
        ylim = (min(map(np.min, signals)), max(map(np.max, signals)))

    fig, axs = plt.subplots(C + hasM, N, sharex="all", sharey="none", squeeze=False)
    for k, (signal, name) in enumerate(zip(signals, arguments)):
        for n in range(signal.shape[0]):
            if signal.shape[1] < C:
                axs[-1, n].plot(signal[n, 0, :], c="k", label=name)
            else:
                c = colores[k % len(colores)]
                for i in range(C):
                    axs[i, n].plot(signal[n, i, :], f"{c}-", label=name)
                    if sharey:
                        axs[i, n].set_ylim(ylim)
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
        y = np.sin(x)
    elif "tri" in symbol:
        y = sawtooth(x, width=0.5)
    elif "saw" in symbol:
        y = sawtooth(x, width=1.0)
    elif "sq" in symbol:
        y = square(x)
    else:
        raise ValueError("unknown symbol")

    _ax.plot(x, y, linewidth=3, c="k")


def plot_signal_heatmap(data, symbols):
    n = len(symbols)
    assert data.shape[0] == n == data.shape[1]

    fig = plt.figure()
    ax = fig.add_axes((0.15, 0.15, 0.7, 0.7))
    ax.imshow(data, norm=colors.SymLogNorm(linthresh=0.03, base=np.e))

    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5)
    ax.grid(which="major", color=ax.get_facecolor(), linestyle='-', linewidth=5)
    ax.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    for i, j in product(range(n), repeat=2):
        col = "black" if data[i, j] > 0 else "white"
        ax.text(j, i, f"{data[i, j]:.3}", ha="center", va="center", color=col)

    pos_tick = np.linspace(0, 1, 2 * n + 1)[1::2]
    size = 1 / n * 1.75

    for i in range(n):
        add_plot_tick(ax, symbols[i], pos=pos_tick[i], where="tensor", size=size)
        add_plot_tick(ax, symbols[i], pos=pos_tick[-i - 1], where="y", size=size)

    return fig
