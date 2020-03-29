#!/usr/bin/env python
from argparse import ArgumentParser
from os import path, makedirs

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from colorama import Fore
from tqdm import tqdm

from thesis import plot
from thesis.data.toy import ToyData
from thesis.io import load_model, get_newest_file, save_append
from thesis.setup import TOY_SIGNALS, DEFAULT_DATA

mpl.use("agg")


def make_cross_likelihood_plot(args):
    fp = f"./figures/{args.basename}/cross_likelihood.npy"
    K = len(TOY_SIGNALS)
    k = TOY_SIGNALS.index(args.k)

    model = load_model(args.weights, args.device).to(args.device)

    test_set = ToyData(
        path=f"{args.data}/test/", mix=False, sources=True, mel_sources=True
    )
    results = np.zeros((K, K, len(test_set)))

    for i, (s, m) in enumerate(tqdm(test_set)):
        s, m = s.unsqueeze(1).to(args.device), m.to(args.device)
        logp, _ = model(s, m)
        results[k, :, i] = logp.mean(-1).squeeze().cpu().numpy()

    if path.exists(fp):
        patch = results
        results = np.load(fp)
        results[k, ...] = patch[k, ...]

    np.save(fp, results)
    print(f"{Fore.YELLOW}Saved {Fore.GREEN}{fp}{Fore.RESET}")


def make_separation_examples(args):
    model = load_model(args.weights, args.device).to(args.device)
    dset = ToyData(path=f"{args.data}/test/", mel=True, sources=True)
    for i, ((mix, mel), sources) in enumerate(tqdm(dset)):
        mix = mix.unsqueeze(0).to(args.device)
        mel = mel.unsqueeze(0).to(args.device)
        ŝ = model.umix(mix, mel)[0]
        _ = plot.toy.reconstruction(sources, ŝ, mix)
        plt.savefig(f"./figures/{args.basename}/separate_{i}.png", dpi=200)
        plt.close()


def make_posterior_examples(args):
    model = load_model(args.weights, args.device)
    dset = ToyData(path=f"{args.data}/test/", mel=True, sources=True)

    for (m, mel), s in tqdm(dset):
        (ŝ,) = model.q_s(m.unsqueeze(0), mel.unsqueeze(0)).mean
        ŝ_mel = torch.cat([model.mel(ŝ[k, :])[None, :] for k in range(4)], dim=0)
        save_append(
            f"./figures/{args.basename}/mean_posterior.pt", (ŝ.unsqueeze(1), ŝ_mel)
        )


def main(args):
    makedirs("./figures", exist_ok=True)
    if args.weights is None:
        match = f"*{args.k}*pt" if args.k else "*pt"
        args.weights = get_newest_file("./checkpoints", match)
        args.basename = path.basename(args.weights)[:-3]
        makedirs(f"./figures/{args.basename}/", exist_ok=True)
    args.device = "cpu" if args.cpu else "cuda"

    with torch.no_grad():
        COMMANDS[args.commands](args)


COMMANDS = {
    "cross-likelihood": make_cross_likelihood_plot,
    "separate": make_separation_examples,
    "posterior": make_posterior_examples,
}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", choices=COMMANDS.keys())
    parser.add_argument("--weights", type=path.abspath)
    parser.add_argument("-k", choices=TOY_SIGNALS)
    parser.add_argument("--data", type=path.abspath, default=DEFAULT_DATA)
    parser.add_argument("-cpu", action="store_true")
    main(parser.parse_args())
