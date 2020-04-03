#!/usr/bin/env python
from argparse import ArgumentParser
from os import path

import numpy as np
from torch.nn import functional as F
import torch
from matplotlib import pyplot as plt

from thesis import plot
from thesis.data.toy import ToyData
from thesis.io import load_model, get_newest_file, exit_prompt
from thesis.setup import TOY_SIGNALS, DEFAULT_DATA
from train import _load_prior_networks
from thesis.functional import mel_spectrogram


def show_sample(args):
    model = load_model(args.weights, args.device)
    model.p_s = _load_prior_networks(prefix="Mar26", device=args.device)

    data = ToyData(
        f"{args.data}/test/", mix=True, mel=True, source=True, rand_amplitude=0.1
    )

    for (m, mel), s in data.loader(1):
        ŝ, m_, log_q_ŝ, p_ŝ, α, β = model.test_forward(m, mel)
        plot.toy.reconstruction(s, ŝ, m, m_)
        plot.toy.reconstruction(m, m_, p_ŝ, ylim=(-500, 1))
        # plot.toy.reconstruction(m, m_, log_q_ŝ)
        # plot.toy.reconstruction(m, m_, α, β)
        # μ_ŝ = model.q_s(m.unsqueeze(0), mel.unsqueeze(0)).mean
        # _ = plot.toy.reconstruction(s, μ_ŝ, m)
        plt.show()
        exit_prompt()


def show_sample_wn(args):
    k = TOY_SIGNALS.index(args.k)
    model = load_model(args.weights, args.device)
    model.p_s = load_model(
        get_newest_file("./checkpoints", f"Mar26*{args.k}*pt"), args.device
    ).to(args.device)

    data = ToyData(f"{args.data}/test/", source=k, mel_source=True, rand_amplitude=0.1)

    for (s, s_mel) in data.loader(1):
        s_noised = (s + 0.1 * torch.randn_like(s)).clamp(-1, 1)
        s_mel_noised = mel_spectrogram(s_noised.squeeze()).unsqueeze(0)
        ŝ = model.forward(s_noised, s_mel_noised)
        l1 = F.l1_loss(ŝ, s, reduction='none')
        mse = F.mse_loss(ŝ, s, reduction='none')
        plot.toy.reconstruction(s, ŝ, l1, mse)
        plt.show()
        exit_prompt()


def show_cross_likelihood():
    log_p = np.load("./figures/cross_likelihood.npy")

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
    model = load_model(args.weights, args.device).to(args.device)
    model.eval()

    data = ToyData(
        f"{args.data}/test/", mix=True, mel=True, source=True, mel_source=True
    )

    for (m, m_mel), (s, s_mel) in data:
        rand_s = torch.rand_like(m) * 0.1
        rand_mel = mel_spectrogram(rand_s)

        sig = torch.cat((s, m, rand_s), dim=0).unsqueeze(1)
        mel = torch.cat((s_mel, m_mel.unsqueeze(0), rand_mel), dim=0)

        log_p, _ = model(sig, mel)

        plot.toy.reconstruction(sig, sharey=False)
        plot.toy.reconstruction(log_p, sharey=False)
        plt.show()
        exit_prompt()


def show_posterior(args):
    model = load_model(args.weights, args.device).to(args.device)
    model.eval()

    data = torch.load(f"./figures/{args.basename}/mean_posterior.pt")

    for s, _ in data:
        s_max = s.detach().squeeze().abs().max(dim=1).values[:, None, None]
        s = s / s_max
        s_mel = mel_spectrogram(s)
        log_p, _ = model(s, s_mel.squeeze())

        plot.toy.reconstruction(s, sharey=False)
        plot.toy.reconstruction(log_p, sharey=False)
        plt.show()
        exit_prompt()


def main(args):
    if args.weights is None:
        match = f"*{args.k}*pt" if args.k else "*pt"
        args.weights = get_newest_file("./checkpoints", match)
        args.basename = path.basename(args.weights)[:-3]
    args.device = "cpu" if not args.gpu else "cuda"

    with torch.no_grad():
        COMMANDS[args.command](args)


COMMANDS = {
    "sample": show_sample,
    "posterior": show_posterior,
    "cross-likelihood": show_cross_likelihood,
    "prior": show_prior,
    "wn": show_sample_wn,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", choices=COMMANDS.keys())
    parser.add_argument("--weights", type=path.abspath)
    parser.add_argument("--data", type=path.abspath, default=DEFAULT_DATA)
    parser.add_argument("-k", choices=TOY_SIGNALS)
    parser.add_argument("-gpu", action="store_true")
    main(parser.parse_args())
