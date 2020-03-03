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


PRINT_LENGTH = 2000


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


def plot_reconstruction(
        S: torch.Tensor, S_tilde: torch.Tensor, m: torch.Tensor = None
) -> plt.Figure:
    if S.ndim == 3:
        S = S.squeeze()
    if S.ndim == 1:
        S = S.unsqueeze(0)
    if S_tilde.ndim == 1:
        S_tilde = S_tilde.unsqueeze(0)
    n = S.shape[0]
    cols = n + 1 if m is not None else n
    fig, axs = plt.subplots(cols, 2, sharex="all")
    if axs.ndim == 1:
        axs = axs[None, ...]
    if S_tilde.ndim == 2:
        for i in range(n):
            axs[i, 0].plot(S[i, 100:PRINT_LENGTH], c="r")
            axs[i, 1].plot(S_tilde[i, 100:PRINT_LENGTH], c="b")
            if m is not None:
                axs[-1, 0].plot(m[0, 100:PRINT_LENGTH], c="g")
    else:
        for i in range(n):
            axs[i, 0].imshow(S[i])
            axs[i, 1].imshow(S_tilde[i])
            axs[-1, 0].imshow(m[0])
    return fig


def plot_one_singal(s: torch.Tensor) -> plt.Figure:
    fig = plt.figure()
    plt.plot(s.squeeze()[100:PRINT_LENGTH])
    return fig


def _tuple_unsequeeze(x):
    if isinstance(x, tuple):
        m = x[0]
        x = (x[0].unsqueeze(0), torch.tensor([x[1]]))
    else:
        m = x
        x = x.unsqueeze(0)
    return m, x


def example_reconstruction(
        model: nn.Module, data: Dataset
) -> Generator[plt.Figure, None, None]:
    for i, (s, _) in enumerate(data):
        s_tilde = model.infer(s.unsqueeze(0)).squeeze()
        s_tilde = s_tilde.argmax(0)
        fig = plot_reconstruction(s, s_tilde)
        yield fig


def z_example_reconstruction(
        model: nn.Module, data: Dataset
) -> Generator[plt.Figure, None, None]:
    for i, (x, y) in enumerate(data):
        x = x.unsqueeze(0)
        z = model(x, y)
        yield plot_one_singal(z)
        x_tilde = model.infer(z)
        yield plot_reconstruction(x, x_tilde)


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
