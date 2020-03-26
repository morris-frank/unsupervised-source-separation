#!/usr/bin/env python
from argparse import ArgumentParser
import torch
from os import makedirs
from os.path import abspath, exists

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore
from tqdm import tqdm

from thesis.data.toy import ToyData
from thesis.io import load_model
from thesis.plot import toy
from thesis.utils import get_newest_file

mpl.use("agg")


TOY_SOURCES = ["sin", "square", "saw", "triangle"]


def make_cross_likelihood_plot(data, _k, weight):
    fp = "./figures/cross_likelihood.npy"
    K = len(TOY_SOURCES)
    k = TOY_SOURCES.index(_k)

    model = load_model(f"{weight}", "cuda").to("cuda")

    test_set = ToyData(path=f"{data}/test/", mix=False, sources=True, mel_sources=True)
    results = np.zeros((K, K, len(test_set)))

    for i, (s, m) in enumerate(tqdm(test_set)):
        s, m = s.to("cuda"), m.to("cuda")
        logp, _ = model(s, m)
        results[k, :, i] = logp.mean(-1).squeeze().cpu().numpy()

    if exists(fp):
        patch = results
        results = np.load(fp)
        results[k, ...] = patch[k, ...]

    np.save(fp, results)
    print(f"{Fore.YELLOW}Saved {Fore.GREEN}{fp}{Fore.RESET}")


def make_separation_examples(data):
    weights = "Mar11-1028_UMixer_supervised_010120.pt"
    makedirs(f"./figures/{weights}/", exist_ok=True)
    model = load_model(f"./checkpoints/{weights}", "cuda").to("cuda")
    dset = ToyData(path=f"{data}/test/", mel=True, sources=True)
    for i, ((mix, mel), sources) in enumerate(tqdm(dset)):
        mix = mix.unsqueeze(0).to("cuda")
        mel = mel.unsqueeze(0).to("cuda")
        ŝ = model.umix(mix, mel)[0]
        _ = toy.reconstruction(sources, ŝ, mix)
        plt.savefig(f"./figures/{weights}/separate_{i}.png", dpi=200)
        plt.close()


def main(args):
    if args.weights is None:
        match = "*pt" if args.k is None else f"*{args.k}*pt"
        args.weights = get_newest_file("./checkpoints", match)
        print(
            f"{Fore.YELLOW}Weights not given. Using instead: {Fore.GREEN}{args.weights}{Fore.RESET}"
        )

    if not exists(args.data):
        raise FileNotFoundError(f"This data directory does not exits, you  stupid fuck\n{args.data}")

    makedirs("./figures", exist_ok=True)

    if args.command == "cross-likelihood":
        with torch.no_grad():
            make_cross_likelihood_plot(args.data, args.k, args.weights)
    elif args.command == "separate":
        make_separation_examples(args.data)
    else:
        raise ValueError("Invalid Command given")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", type=str, help="show ∨ something")
    parser.add_argument("--weights", type=abspath)
    parser.add_argument("-k", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data", type=abspath, default="/home/morris/var/data/toy")
    main(parser.parse_args())
