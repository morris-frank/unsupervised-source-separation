#!/usr/bin/env python
from argparse import ArgumentParser
from itertools import product, combinations
from os import path

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from torchaudio.functional import istft

from thesis import plot
from thesis.data.toy import ToyData
from thesis.io import load_model, exit_prompt, get_newest_checkpoint, get_newest_file
from thesis.nn.modules import MelSpectrogram
from thesis.setup import TOY_SIGNALS, DEFAULT_TOY, MUSDB_SIGNALS

mpl.use("TkCairo")


def show_sample(args):
    model = load_model(args.weights, args.device)
    model.p_s = [
        load_model(get_newest_checkpoint("*Discr*"), args.device).to(args.device)
    ]

    data = ToyData(
        args.data, "test", mix=True, mel=True, source=True, rand_amplitude=0.1
    )

    for (m, mel), s in data.loader(1):
        ŝ = model.forward(m, mel)
        plot.toy.reconstruction(s, ŝ, m, ŝ.mean(dim=1))

        plt.show()
        exit_prompt()


def show_sample_denoiser(args):
    k = TOY_SIGNALS.index(args.k)
    model = load_model(args.weights, args.device)
    model.p_s = [
        load_model(get_newest_checkpoint(f"Apr04*{args.k}*pt"), args.device).to(
            args.device
        )
    ]

    data = ToyData(args.data, "test", source=k, rand_amplitude=0.1)

    for s in data.loader(1):
        s_noised = (s + 0.01 * torch.randn_like(s)).clamp(-1, 1)
        ŝ = model.forward(s_noised)

        plot.toy.reconstruction(s_noised, s, ŝ)
        plt.show()
        exit_prompt()


def show_cross_likelihood(args):
    log_p = np.load(get_newest_file("./figures", f"{args.k}*cross_likelihood.npy"))
    log_p[log_p == -np.inf] = -1e3
    log_p = np.maximum(log_p, -1e3)
    log_p = log_p.swapaxes(0, 1)

    n = (log_p[0, 0, :] != 0).sum()

    # fig = plt.figure()
    fig, (ax) = plt.subplots(
        1, 1, gridspec_kw=dict(left=0.1, right=0.95, top=0.9, bottom=0.05, wspace=0.2)
    )

    # ax = fig.add_axes((0.15, 0.15, 0.7, 0.7))
    plot.toy.plot_signal_heatmap(ax, log_p[:, :, :n].mean(-1), TOY_SIGNALS)
    # ax1.set_title(r"mean of likelihood log p(s)")

    # ax = fig.add_axes((0.15, 0.15, 0.7, 0.7))
    # plot.toy.plot_signal_heatmap(ax2, log_p.var(-1), MUSDB_SIGNALS)
    # ax2.set_title("var of likelihood log p(s)")

    # plt.savefig('example.pgf', dpi=80, bbox_inches=Bbox([[1, 0], [6, 5]]), transparent=True)

    # tikzplotlib.clean_figure()
    # tikzplotlib.save('example.tex')

    fig.show()

    import ipdb

    ipdb.set_trace()

    exit_prompt()
    plt.close()


def show_prior(args):
    model = load_model(args.weights, args.device)

    data = ToyData(args.data, "test", mix=True, mel=True, source=True, mel_source=True)
    mel_spectr = MelSpectrogram()

    for (m, m_mel), (s, s_mel) in data:
        rand_s = 0.0 * 2 * torch.rand((1, 3072)) - 1
        rand_mel = mel_spectr(rand_s)

        sig = torch.cat((s, m, rand_s), dim=0).unsqueeze(1).repeat(1, 4, 1)
        mel = torch.cat((s_mel, m_mel.unsqueeze(0), rand_mel), dim=0).repeat(1, 4, 1)

        t_s = model(sig, mel)
        s, y = sig.transpose(0, 1), t_s.transpose(0, 1)
        plot.toy.reconstruction(s, y, sharey=False)

        plt.show()
        exit_prompt()


def show_prior_z(args):
    model = load_model(args.weights, args.device)
    _istft = lambda x: istft(x, n_fft=128, length=3072, normalized=True)

    while True:
        z = torch.rand((1, 4 * model.params["kwargs"]["in_channel"], 3072))
        out = model.reverse(z).view(4, -1, 3072)

        # sgrams = out.view(4, -1, 2, 3072).permute(0, 1, 3, 2)
        # waveforms = [_istft(sgrams[i, ...]) for i in range(4)]

        _, axs = plt.subplots(4)
        a, b = 1000, 2000
        for i in range(4):
            axs[i].plot(waveforms[i][a : a + b])
        plt.show()
        exit_prompt()


def show_hist_posterior():
    from thesis.audio import oscillator

    mpl.style.use(f"./thesis/plot/mpl.style")
    _, axs = plt.subplots(4)
    for i, k in enumerate(["sin", "square", "saw", "triangle"]):
        hist = torch.histc(
            torch.tensor(oscillator(10000, k, 500, 0)[0])
            + (torch.rand(10000) - 0.5) * 0.04,
            bins=20,
            min=-1,
            max=1,
        )
        axs[i].plot(hist, linewidth=4)
        axs[i].grid(b=None)
        axs[i].tick_params(axis="both", labelsize=0)
    plt.show()


def show_discrprior_roc(args):
    res = np.load("figures/Jun14_prior_cross_entropy.npy", allow_pickle=True).item()
    ŷ = np.array(res["ŷ"])
    y = np.array(res["y"])
    ŷ = ŷ / ŷ.sum(-1, keepdims=True)
    true_class_ŷ = ŷ[np.indices(y.shape)[0], y]
    cmat = metrics.confusion_matrix(y, ŷ.argmax(-1))

    fig, ax = plt.subplots(
        1, 1, gridspec_kw=dict(left=0.1, right=0.95, top=0.9, bottom=0.05, wspace=0.2)
    )
    plot.toy.plot_signal_heatmap(ax, cmat, MUSDB_SIGNALS)
    plt.show()


def show_posterior(args):
    model = load_model(args.weights, args.device)
    mel_spectr = MelSpectrogram()

    data = torch.load(f"./figures/{args.basename}/mean_posterior.pt")

    for s, _ in data:
        s_max = s.detach().squeeze().abs().max(dim=1).values[:, None, None]
        s = s / s_max
        s_mel = mel_spectr(s)
        log_p, _ = model(s, s_mel.squeeze())

        plot.toy.reconstruction(s, sharey=False)
        plot.toy.reconstruction(log_p, sharey=False)
        plt.show()
        exit_prompt()
        plt.close()


def show_noise_plot(args):
    pref = f"*{args.k}*" if args.k is not None else "**"
    df = np.load(
        get_newest_file("./figures", f"{pref}/noise_likelihood.npy"), allow_pickle=True,
    ).item()

    l = []
    for σ, (i, k) in product(df.keys(), enumerate(TOY_SIGNALS)):
        l.extend([(σ, k, v) for v in df[σ][i].tolist()])

    df = pd.DataFrame(l, columns=["noise-level", "signal", "log-likelihood"])
    df = df[df["log-likelihood"] != 0]
    df = df[df["noise-level"] != 0.001]
    # df = df[df['signal'] == 'triangle']

    _, ax = plt.subplots()
    # plot.toy.add_plot_tick(ax, symbol=args.k, size=0.1)
    sns.boxplot(
        x="noise-level",
        y="log-likelihood",
        hue="signal",
        data=df,
        ax=ax,
        showfliers=False,
    )
    ax.set(yscale="log")
    plt.show()


def show_mel(args):
    """
    Shows the Mel-spectrograms as they are gonna be computed for the toy data set.
    """
    from thesis.data.musdb import MusDB

    # data = ToyData(f"{args.data}/test/", source=True, mel_source=True)
    data = MusDB(f"/home/morris/var/data/musdb18", subsets="train", mel=True)
    signals = ["mix", "drums", "bass", "other", "vocals"]

    for _, m in data:
        fig, axs = plt.subplots(5)
        for i, ax in enumerate(axs):
            ax.imshow(m[i, :, ::30])
            ax.set_title(signals[i])
        plt.show()
        exit_prompt()
        plt.close(fig)


def show_interpolate_prior(args):
    length = 16_384
    model = load_model(args.weights, args.device)

    if "time" in model.name:
        opt = {"source": True}
    else:
        opt = {"mel_source": True}

    dset = ToyData(
        args.data, "test", noise=0.0, rand_amplitude=0.2, length=length, **opt
    ).loader(1)
    for a, b in combinations(dset, 2):
        α, *_ = model.forward(a)
        β, *_ = model.forward(b)
        γ = (α + β) / 2

        c = model.reverse(γ)
        np.save(
            "../thesis-tex/data/prior_toy_interpolate.npy", torch.cat((a, b, c)).numpy()
        )
        fig = plot.toy.reconstruction(a, b, c, sharey=True, ylim=[-1, 1])
        plt.show()
        plt.close(fig)


def show_data(args):
    test_set = ToyData(
        args.data,
        "test",
        noise=0.1,
        rand_amplitude=0.2,
        length=1000,
        source=True,
    )
    for s in test_set:
        fig = plot.toy.reconstruction(s, sharey=True, ylim=[-1, 1])
        plt.show()
        plt.close(fig)


def main(args):
    # if args.weights is None:
    #     args.weights = get_newest_checkpoint(f"*{args.k}*pt" if args.k else "*pt")
    #     args.basename = path.basename(args.weights)[:-3]
    args.device = "cpu" if not args.gpu else "cuda"

    with torch.no_grad():
        COMMANDS[args.command](args)


COMMANDS = {
    "sample": show_sample,
    "posterior": show_posterior,
    "channels": show_cross_likelihood,
    "prior": show_prior,
    "prior-z": show_prior_z,
    "denoiser": show_sample_denoiser,
    "noise": show_noise_plot,
    "mel": show_mel,
    "discrprior_roc": show_discrprior_roc,
    "interpolate-prior": show_interpolate_prior,
    "data": show_data,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", choices=COMMANDS.keys())
    parser.add_argument("--weights", type=get_newest_checkpoint)
    parser.add_argument("--data", type=path.abspath, default=DEFAULT_TOY)
    parser.add_argument("-k", type=str)
    parser.add_argument("-gpu", action="store_true")
    main(parser.parse_args())
