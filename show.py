#!/usr/bin/env python
from argparse import ArgumentParser
from os.path import abspath

import numpy as np
import torch
from colorama import Fore
from matplotlib import pyplot as plt

from thesis import plot
from thesis.data.toy import ToyDataSourceK
from thesis.io import load_model
from thesis.utils import get_newest_file


def show_cross_likelihood():
    data = np.load("./figures/cross_likelihood.npy")
    log_p = data[..., 0]

    fig = plot.toy.plot_signal_heatmap((log_p.mean(-1)), ["sin", "sq", "saw", "tri"])
    fig.suptitle(r"mean of likelihood log p(s)")
    fig.show()
    input("?")
    fig = plot.toy.plot_signal_heatmap(log_p.var(-1), ["sin", "sq", "saw", "tri"])
    fig.suptitle("var of likelihood log p(s)")
    fig.show()
    input("?")
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

        print(f'log_p|log_det: {log_p.detach().mean().item()}|{logdet.detach().item()}\tlog_p_|log_det_: {log_p_.detach().mean().item()}|{logdet_.detach().item()}')
        _ = plot.toy.plot_reconstruction(torch.cat([s, s_], 1), torch.cat([log_p, log_p_], 1))
        plt.show()
        input()


def main(args):
    if args.weights is None:
        args.weights = get_newest_file("./checkpoints")
        print(
            f"{Fore.YELLOW}Weights not given. Using instead: {Fore.GREEN}{args.weights}{Fore.RESET}"
        )

    if args.command == "sample":
        from thesis.plot.toy import example_reconstruction

        model = load_model(args.weights, args.device)
        model.eval()
        data = map_dataset(model, args.data, "test")

        with torch.no_grad():
            for fig in example_reconstruction(model, data):
                fig.show()
                input()
                plt.close(fig)

    elif args.command == "z-sample":
        from thesis.plot.toy import z_example_reconstruction

        model = load_model(args.weights, args.device)
        model.eval()
        data = map_dataset(model, args.data, "test")
        with torch.no_grad():
            for fig in z_example_reconstruction(model, data):
                fig.show()
                input("?")
                plt.close(fig)

    elif args.command == "rand-z":
        model = load_model(args.weights, args.device)
        model.eval()
        with torch.no_grad():
            while True:
                z = torch.rand(1, 1, 2 ** 11)
                # z = torch.zeros(1, 1, 2**11)
                # z.fill_(torch.rand(1).item())
                m = model.infer(z)
                m.clamp_(-1, 1)
                fig = plot.toy.plot_one_singal(m)
                fig.show()
                input("?")
                plt.close()

    elif args.command == "cross-likelihood":
        show_cross_likelihood()

    elif args.command == "logpz":
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
