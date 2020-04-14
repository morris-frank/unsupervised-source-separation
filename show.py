#!/usr/bin/env python
from functools import partial
from argparse import ArgumentParser
from os import path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from thesis import plot
from thesis.data.toy import ToyData
from thesis.io import load_model, exit_prompt, get_newest_checkpoint, get_newest_file
from thesis.nn.modules import MelSpectrogram
from thesis.setup import TOY_SIGNALS, DEFAULT_DATA
from train import _load_prior_networks


def show_sample(args):
    model = load_model(args.weights, args.device)
    # model.p_s = _load_prior_networks(device=args.device)
    model.p_s = [load_model(get_newest_checkpoint('*Discr*'), args.device).to(args.device)]

    data = ToyData(
        f"{args.data}/test/", mix=True, mel=True, source=True, rand_amplitude=0.1
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

    data = ToyData(f"{args.data}/test/", source=k, rand_amplitude=0.1)

    for s in data.loader(1):
        s_noised = (s + 0.01 * torch.randn_like(s)).clamp(-1, 1)
        ŝ = model.forward(s_noised)

        plot.toy.reconstruction(s_noised, s, ŝ)
        plt.show()
        exit_prompt()


def show_cross_likelihood(args):
    log_p = np.load(get_newest_file('./figures', '*_cross_likelihood.npy'))

    fig = plot.toy.plot_signal_heatmap((log_p.mean(-1)), TOY_SIGNALS)
    fig.suptitle(r"mean of likelihood log p(s)")
    fig.show()
    exit_prompt()

    fig = plot.toy.plot_signal_heatmap(log_p.var(-1), TOY_SIGNALS)
    fig.suptitle("var of likelihood log p(s)")
    fig.show()
    exit_prompt()
    plt.close()


def show_prior(args):
    model = load_model(args.weights, args.device)

    data = ToyData(
        f"{args.data}/test/", mix=True, mel=True, source=True, mel_source=True
    )
    mel_spectr = MelSpectrogram()
    _rand = torch.rand((1, 3072))
    rand = lambda: 2*_rand-1

    for (m, m_mel), (s, s_mel) in data:
        rand_s = 0.1 * rand()
        rand_mel = mel_spectr(rand_s)

        sig = torch.cat((s, m, rand_s), dim=0).unsqueeze(1).repeat(1, 4, 1)
        mel = torch.cat((s_mel, m_mel.unsqueeze(0), rand_mel), dim=0).repeat(1, 4, 1)

        t_s = model(sig, mel)
        s, y = sig.transpose(0, 1), t_s.transpose(0, 1)
        plot.toy.reconstruction(s, y, sharey=False)

        plt.show()
        exit_prompt()


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
    df = pd.DataFrame(
        np.load(
            get_newest_file("./figures", f"**{args.k}*/noise_likelihood.npy"),
            allow_pickle=True,
        ).item()
    )
    df = df.melt(var_name="amount of noise", value_name="log-likelihood under prior")

    _, ax = plt.subplots()
    plot.toy.add_plot_tick(ax, symbol=args.k, size=0.1)
    sns.boxplot(x="amount of noise", y="log-likelihood under prior", data=df, ax=ax)
    plt.show()


def show_gan(args):
    model = load_model(args.weights, args.device)
    model.p_s = [
        load_model(get_newest_checkpoint(f"Apr06*{args.k}*pt"), args.device).to(
            args.device
        )
    ]

    while True:
        z = torch.randn((1, 1, 3072))
        ŝ, log_p = model.forward(z)
        plot.toy.reconstruction(z, ŝ, log_p)
        plt.show()
        exit_prompt()
        plt.close()


def show_mel(args):
    """
    Shows the Mel-spectrograms as they are gonna be computed for the toy data set.
    """
    data = ToyData(f"{args.data}/test/", source=True, mel_source=True)

    for s, m in data:
        fig, axs = plt.subplots(2, 2)
        axs = axs.flatten()
        for i in range(4):
            axs[i].imshow(m[i, ...])
        plt.show()
        exit_prompt()
        plt.close(fig)


def main(args):
    if args.weights is None:
        args.weights = get_newest_checkpoint(f"*{args.k}*pt" if args.k else "*pt")
        args.basename = path.basename(args.weights)[:-3]
    args.device = "cpu" if not args.gpu else "cuda"

    with torch.no_grad():
        COMMANDS[args.command](args)


COMMANDS = {
    "sample": show_sample,
    "posterior": show_posterior,
    "cross-likelihood": show_cross_likelihood,
    "prior": show_prior,
    "denoiser": show_sample_denoiser,
    "noise": show_noise_plot,
    "mel": show_mel,
    "gan": show_gan,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", choices=COMMANDS.keys())
    parser.add_argument("--weights", type=get_newest_checkpoint)
    parser.add_argument("--data", type=path.abspath, default=DEFAULT_DATA)
    parser.add_argument("-k", choices=TOY_SIGNALS)
    parser.add_argument("-gpu", action="store_true")
    main(parser.parse_args())
