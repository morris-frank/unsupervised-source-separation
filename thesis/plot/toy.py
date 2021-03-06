from itertools import product
from math import tau as τ

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import colors
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import sawtooth, square

from ..utils import get_func_arguments

PRINT_START = 500
PRINT_LENGTH = 3000


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

    fig, axs = plt.subplots(C + hasM, N, sharex="all", sharey="none", squeeze=False, figsize=(24, 7))
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


def add_plot_tick(
    ax: plt.Axes, symbol: str, pos: float = 0.5, where: str = "x", size: float = 0.05
):

    if "x" in where:
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
        _ax.plot(x, y, linewidth=3, c="k")
    elif "tri" in symbol:
        y = sawtooth(x, width=0.5)
        _ax.plot(x, y, linewidth=3, c="k")
    elif "saw" in symbol:
        y = sawtooth(x, width=1.0)
        _ax.plot(x, y, linewidth=3, c="k")
    elif "sq" in symbol:
        y = square(x)
        _ax.plot(x, y, linewidth=3, c="k")
    elif symbol in ["drums", "bass", "vocals", "other"]:
        sf = "_grey" if rcParams["figure.facecolor"] == "#f7f7f7" else ""
        icon = plt.imread(f"thesis/plot/{symbol}{sf}.png")
        _ax.imshow(np.repeat(icon[..., None], 3, 2))
    else:
        raise ValueError("unknown symbol")


def plot_signal_heatmap(ax, data, symbols):
    n = len(symbols)
    assert data.shape[0] == n == data.shape[1]

    cmap = sns.light_palette("#99961a")
    # ax.imshow(data, norm=colors.SymLogNorm(linthresh=0.03, base=np.e))
    sns.heatmap(data, ax=ax, annot=True, linewidths=2, cbar=False, square=True, norm=colors.SymLogNorm(linthresh=0.03, base=4*np.e), cmap=cmap)

    # for edge, spine in ax.spines.items():
    #     spine.set_visible(False)
    # ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5)
    # ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5)
    # ax.grid(which="major", color=ax.get_facecolor(), linestyle="-", linewidth=5)
    ax.tick_params(
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )

    # for i, j in product(range(n), repeat=2):
    #     col = "black" if data[i, j] > 0 else "white"
    #     t = f"{data[i, j]:.3}" if isinstance(data[i, j], float) else f"{data[i, j]}"
    #     ax.text(j, i, t, ha="center", va="center", color=col)

    pos_tick = np.linspace(0, 1, 2 * n + 1)[1::2]
    size = 1 / n * 1.75

    for i in range(n):
        add_plot_tick(ax, symbols[i], pos=pos_tick[i], where="x", size=size)
        add_plot_tick(ax, symbols[i], pos=pos_tick[-i - 1], where="y", size=size)
