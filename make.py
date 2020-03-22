#!/usr/bin/env python
import asyncio
from argparse import ArgumentParser
from os import makedirs
from os.path import abspath

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore
from tqdm import tqdm

from thesis.data.toy import ToyDataMixes, ToyDataSources
from thesis.io import load_model
from thesis.plot import toy
from thesis.utils import get_newest_file

mpl.use("agg")


async def make_cross_likelihood_plot(data):
    fp = "./figures/cross_likelihood.npy"
    K = 4

    weights = [
        "Mar22-0051_Flowavenet_sin_049999.pt",
        "Mar22-0051_Flowavenet_square_049999.pt",
        "Mar22-0051_Flowavenet_saw_049999.pt",
        "Mar22-0051_Flowavenet_triangle_049999.pt",
    ]

    models = [load_model(f"./checkpoints/{w}", "cuda").to("cuda") for w in weights]

    test_set = ToyDataSources(path=f"{data}/test/", mel=True)
    results = np.zeros((K, K, len(test_set), 2))

    async def infer(k, idx, sign, mel):
        results[k, :, idx, :] = models[k](sign, mel)

    for i, (s, m) in enumerate(tqdm(test_set)):
        s, m = s.to("cuda"), m.to("cuda")
        await asyncio.gather(*(infer(k, i, s, m) for k in range(K)))

    np.save(fp, results)


def make_separation_examples(data):
    weights = "Mar11-1028_UMixer_supervised_010120.pt"
    makedirs(f"./figures/{weights}/", exist_ok=True)
    model = load_model(f"./checkpoints/{weights}", "cuda").to("cuda")
    dset = ToyDataMixes(path=f"{data}/test/", mel=True, sources=True)
    for i, ((mix, mel), sources) in enumerate(tqdm(dset)):
        mix = mix.unsqueeze(0).to("cuda")
        mel = mel.unsqueeze(0).to("cuda")
        ŝ = model.umix(mix, mel)[0]
        _ = toy.plot_reconstruction(sources, ŝ, mix)
        plt.savefig(f"./figures/{weights}/separate_{i}.png", dpi=200)
        plt.close()


def main(args):
    if args.weights is None:
        args.weights = get_newest_file("./checkpoints")
        print(
            f"{Fore.YELLOW}Weights not given. Using instead: {Fore.GREEN}{args.weights}{Fore.RESET}"
        )

    makedirs("./figures", exist_ok=True)

    if args.command == "cross-likelihood":
        make_cross_likelihood_plot(args.data)
    elif args.command == "separate":
        asyncio.run(make_separation_examples(args.data))
    else:
        raise ValueError("Invalid Command given")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", type=str, help="show ∨ something")
    parser.add_argument("--weights", type=abspath)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data", type=abspath, default="/home/morris/var/data/toy")
    main(parser.parse_args())
