#!/usr/bin/env python
from argparse import ArgumentParser
from os.path import abspath

import numpy as np
import torch
from colorama import Fore
from matplotlib import pyplot as plt

from thesis import plot
from thesis.data.toy import TOY_SIGNALS
from thesis.data.toy import ToyData
from thesis.io import load_model, get_newest_file, exit_prompt
from train import _load_prior_networks


def show_sample(data, weights):
    model = load_model(weights, "cpu")
    model.p_s = _load_prior_networks(prefix="Mar26", device="cpu")

    # dset = ToyDataRandomAmplitude(path=f"{data}/test/")
    dset = ToyData(path=f"{data}/test/", mel=True, sources=True)

    for (m, mel), s in dset:
        ŝ, m_, log_q_ŝ, α, β = model.test_forward(m.unsqueeze(0), mel.unsqueeze(0))
        plot.toy.reconstruction(s, ŝ, m, m_)
        # plot.toy.reconstruction(m, m_, p_ŝ)
        plot.toy.reconstruction(m, m_, log_q_ŝ)
        plot.toy.reconstruction(m, m_, α, β)
        # μ_ŝ = model.q_s(m.unsqueeze(0), mel.unsqueeze(0)).mean
        # _ = plot.toy.reconstruction(s, μ_ŝ, m)
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


def make_mel(signal):
    from torchaudio.transforms import MelSpectrogram

    mel_channels = 80
    n_fft = 1024
    hop_length = 256
    sr = 16000
    mel = MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=mel_channels,
        f_min=125,
        f_max=7600,
    )
    return mel(signal)


def show_prior(data, source):
    weights = get_newest_file("./checkpoints", f"*{source}*pt")
    model = load_model(weights, "cpu").to("cpu")
    model.eval()

    dset = ToyData(path=f"{data}/test/", mel=True, sources=True, mel_sources=True)

    for (m, m_mel), (s, s_mel) in dset:
        rand_s = torch.rand_like(m) * 0.1
        rand_mel = make_mel(rand_s)

        sig = torch.cat((s, m, rand_s), dim=0).unsqueeze(1)
        mel = torch.cat((s_mel, m_mel.unsqueeze(0), rand_mel), dim=0)

        log_p, _ = model(sig, mel)

        plot.toy.reconstruction(sig, sharey=False)
        plot.toy.reconstruction(log_p, sharey=False)
        plt.show()
        exit_prompt()


def show_posterior(source):
    weights = get_newest_file("./checkpoints", f"*{source}*pt")
    model = load_model(weights, "cpu").to("cpu")
    model.eval()

    data = torch.load("./mean_posterior.pt")

    for s, _ in data:
        s_max = s.detach().squeeze().abs().max(dim=1).values[:, None, None]
        s = s / s_max
        s_mel = make_mel(s)
        log_p, _ = model(s, s_mel.squeeze())

        plot.toy.reconstruction(s, sharey=False)
        plot.toy.reconstruction(log_p, sharey=False)
        plt.show()
        exit_prompt()


def main(args):
    if args.weights is None:
        args.weights = get_newest_file("./checkpoints")
        print(
            f"{Fore.YELLOW}Weights not given. Using instead: {Fore.GREEN}{args.weights}{Fore.RESET}"
        )

    if args.command == "sample":
        with torch.no_grad():
            show_sample(args.data, args.weights)

    elif args.command == "posterior":
        with torch.no_grad():
            show_posterior(args.k)

    elif args.command == "cross-likelihood":
        show_cross_likelihood()

    elif args.command == "prior":
        with torch.no_grad():
            show_prior(args.data, args.k)

    else:
        raise ValueError("Invalid command given")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("--weights", type=abspath)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data", type=abspath, default="/home/morris/var/data/toy")
    parser.add_argument("-k", type=str)
    main(parser.parse_args())
