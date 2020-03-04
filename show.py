#!/usr/bin/env python
from argparse import ArgumentParser
from os.path import abspath

import torch
from colorama import Fore
from matplotlib import pyplot as plt

from thesis import plot
from thesis.io import load_model
from thesis.utils import get_newest_file


def main(args):
    if args.weights is None:
        args.weights = get_newest_file("./checkpoints")
        print(f'{Fore.YELLOW}Weights not given. Using instead: {Fore.GREEN}{args.weights}{Fore.RESET}')

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
                z = torch.rand(1, 1, 2**11)
                #z = torch.zeros(1, 1, 2**11)
                #z.fill_(torch.rand(1).item())
                m = model.infer(z)
                m.clamp_(-1, 1)
                fig = plot.toy.plot_one_singal(m)
                fig.show()
                input("?")
                plt.close()

    elif args.command == "cross-likelihood":
        import numpy as np
        data = np.load('./figures/cross_likelihood.npy')
        log_p = data[..., 0]
        #log_det = data[..., 1]

        fig = plot.toy.plot_signal_heatmap(np.exp(log_p.mean(-1)), ['sin', 'sq', 'saw', 'tri'])
        fig.suptitle(r'exp of mean of log p(s)')
        fig.show()
        input("?")
        fig = plot.toy.plot_signal_heatmap(np.exp(log_p).var(-1), ['sin', 'sq', 'saw', 'tri'])
        fig.suptitle('var of exp of log p(s)')
        fig.show()
        input("?")
        plt.close()

    else:
        raise ValueError("Invalid command given")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("--weights", type=abspath)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data", type=abspath, default="/home/morris/var/data/toy")
    main(parser.parse_args())
