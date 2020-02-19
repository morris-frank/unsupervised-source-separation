#!/usr/bin/env python
from argparse import ArgumentParser
from glob import glob
from os.path import abspath, getmtime
from colorama import Fore

import torch
from matplotlib import pyplot as plt

from thesis.data.wrapper import map_dataset
from thesis.io import load_model


def get_newest_file(folder):
    return sorted(glob(f"{folder}/*pt"), key=lambda x: getmtime(x))[-1]


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

    else:
        raise ValueError("Invalid command given")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("--weights", type=abspath)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data", type=abspath)
    main(parser.parse_args())
