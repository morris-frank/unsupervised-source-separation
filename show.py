#!/usr/bin/env python
from argparse import ArgumentParser
from os.path import abspath

import numpy as np
import torch
from colorama import Fore
from matplotlib import pyplot as plt

from thesis import plot
from thesis.data.toy import ToyDataSourceK, ToyDataRandomAmplitude
from thesis.functional import discretize
from thesis.io import load_model, get_newest_file, exit_prompt
from thesis.data.toy import TOY_SIGNALS
from train import _load_prior_networks


def show_sample(data, weights):
    model = load_model(weights, "cpu")
    model.p_s = _load_prior_networks(prefix='Mar26', device="cpu")

    dset = ToyDataRandomAmplitude(path=f"{data}/test/")

    for (m, mel), s in dset:
        ŝ, m_, p_ŝ, log_q_ŝ = model.test_forward(m.unsqueeze(0), mel.unsqueeze(0))
        plot.toy.reconstruction(s, ŝ, m, m_)
        plot.toy.reconstruction(s, torch.cat(p_ŝ, 1))
        # μ_ŝ = model.q_s(m.unsqueeze(0), mel.unsqueeze(0)).mean
        # μ_ŝ, _ = model.q_s(m.unsqueeze(0), mel.unsqueeze(0))  # For Gaussian
        #s = discretize(s, model.μ).long()
        # μ_ŝ = model.q_s(m.unsqueeze(0), mel.unsqueeze(0)).logits.argmax(
        #     dim=-1
        # )  # For categorical
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


def show_log_p_z_test(data):
    weights = "Mar22-0051_Flowavenet_sin_009425.pt"
    model = load_model(f"./checkpoints/{weights}", "cpu")
    dset = ToyDataSourceK(path=f"{data}/test/", k=0, mel=True)
    dset_alt = ToyDataSourceK(path=f"{data}/test/", k=1, mel=True)
    model.eval()
    for i, ((s, m), (sz, mz)) in enumerate(zip(dset, dset_alt)):
        s[:, 1100:] = sz[:, 1100:]
        m[:, 5:] = mz[:, 5:]
        s, m = s.unsqueeze(0), m.unsqueeze(0)
        log_p, logdet = model(s, m)

        s_ = s * torch.rand_like(s)
        log_p_, logdet_ = model(s_, m)

        print(
            f"log_p|log_det: {log_p.detach().mean().item()}|{logdet.detach().item()}\tlog_p_|log_det_: {log_p_.detach().mean().item()}|{logdet_.detach().item()}"
        )
        _ = plot.toy.reconstruction(
            torch.cat([s, s_], 1), torch.cat([log_p, log_p_], 1)
        )
        plt.show()
        input()


def main(args):
    if args.weights is None:
        args.weights = get_newest_file("./checkpoints")
        print(
            f"{Fore.YELLOW}Weights not given. Using instead: {Fore.GREEN}{args.weights}{Fore.RESET}"
        )

    if args.command == "sample":
        with torch.no_grad():
            show_sample(args.data, args.weights)

    elif args.command == "cross-likelihood":
        show_cross_likelihood()

    elif args.command == "logpz":
        with torch.no_grad():
            show_log_p_z_test(args.data)

    else:
        raise ValueError("Invalid command given")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("--weights", type=abspath)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data", type=abspath, default="/home/morris/var/data/toy")
    main(parser.parse_args())
