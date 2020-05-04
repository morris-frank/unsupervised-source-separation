#!/usr/bin/env python
from argparse import ArgumentParser
from os import path, makedirs, getpid

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from colorama import Fore
from tqdm import tqdm, trange

from thesis import plot
from thesis.data.toy import ToyData, generate_toy
from thesis.data.musdb import MusDBSamples
from thesis.io import load_model, save_append, get_newest_checkpoint, FileLock
from thesis.nn.modules import MelSpectrogram
from thesis.setup import DEFAULT

mpl.use("agg")


def make_noise_likelihood_plot(args):
    k = DEFAULT.signals.index(args.k)
    model = load_model(args.weights, args.device)
    mel = MelSpectrogram()
    data = ToyData(args.data, "test", source=k, mel_source=True)

    results = {}
    for σ in [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]:
        results[σ] = np.zeros((len(data)))
        for i, (s, m) in enumerate(tqdm(data.loader(1))):
            s = (s + σ * torch.randn_like(s)).clamp(-1, 1)
            m = mel(s[0])
            s, m = s.to(args.device), m.to(args.device)
            logp = model(s, m)[0]
            results[σ][i] = logp.mean(-1).item()

    np.save(
        f"./figures/{args.basename}/noise_likelihood.npy", results, allow_pickle=True
    )


def make_cross_likelihood_plot(args):
    weights = get_newest_checkpoint(f"*Flowavenet*pt")
    fp = f"./figures/{path.basename(weights).split('-')[0]}_prior_cross_likelihood.npy"

    print(f"{Fore.YELLOW}I'am building: {Fore.BLUE}{fp}{Fore.RESET}")

    model = load_model(weights, args.device)

    batch_size = 50
    if args.musdb:
        data = MusDBSamples(args.data, "test").loader(batch_size, drop_last=True)
    else:
        data = ToyData(
            args.data, "test", source=True, mel_source=True, interpolate=True
        ).loader(batch_size, drop_last=True)

    K = len(DEFAULT.signals)
    results = np.zeros((K, K, len(data) * batch_size))

    for i, (_, m) in enumerate(tqdm(data)):
        _, _, C, L = m.shape
        m = m.view(batch_size * K, C, L)
        m = m.repeat(1, K, 1).to(args.device)
        logp = model(m)[0]
        results[:, :, i:i+batch_size] = (
            logp.mean(-1)
            .view(batch_size, K, K)
            .permute(1, 2, 0)
            .squeeze()
            .cpu()
            .numpy()
        )

    print(Fore.YELLOW + "Saving to " + fp + Fore.RESET)
    np.save(fp, results)


def make_separation_examples(args):
    model = load_model(args.weights, args.device)
    data = ToyData(args.data, "test", mix=True, mel=True, source=True)
    for i, ((mix, mel), sources) in enumerate(tqdm(data.loader(1))):
        mix, mel = mix.to(args.device), mel.to(args.device)
        ŝ = model.umix(mix, mel)[0]
        _ = plot.toy.reconstruction(sources, ŝ, mix)
        plt.savefig(f"./figures/{args.basename}/separate_{i}.png", dpi=200)
        plt.close()


def make_posterior_examples(args):
    model = load_model(args.weights, args.device)
    dset = ToyData(args.data, "test", mix=True, mel=True, source=True)

    for (m, mel), s in tqdm(dset):
        (ŝ,) = model.q_s(m.unsqueeze(0), mel.unsqueeze(0)).mean
        ŝ_mel = torch.cat([model.mel(ŝ[k, :])[None, :] for k in range(4)], dim=0)
        save_append(
            f"./figures/{args.basename}/mean_posterior.pt", (ŝ.unsqueeze(1), ŝ_mel)
        )


def make_toy_dataset(args):
    length, ns = 3072, 4
    config = {"train": 5_000, "test": 500}

    for name, n in config.items():
        print(f"Generate Toy [{name}] n={n}, length={length}, ns={ns}")
        print(f"Save to {args.data}/{name}")
        for i in trange(n):
            item = generate_toy(length, ns)
            np.save(f"{args.data}/{name}/{name}_{i:05}.npy", item)


def make_musdb_pre_save(args):
    from thesis.data.musdb import MusDB

    n = 100

    data = MusDB(args.data, subsets="test", mel=True)
    data.pre_save(n)


def make_data_distribution(args):
    from thesis.data.musdb import MusDB

    # h = torch.load('musdb_histograms.pt')
    # h = np.mean(h, axis=0)
    # _, axs = plt.subplots(4)
    # for i, ax in zip(range(4), axs):
    #     ax.plot(h[i, :])
    # plt.show()

    fp = "musdb_histograms.pt"
    hists = np.zeros((4, 100))
    bins = np.linspace(-1, 1, 101)
    data = MusDB(args.data, subsets="train")
    n = 10
    for i, track in enumerate(data):
        for k in range(4):
            hists[k, :] += np.histogram(track[k, :], bins=bins)[0] / (data.L * n)

        if i % 10 == 0:
            print(f"{getpid()}: {i}")
            with FileLock(fp):
                save_append(fp, hists)
            hists = np.zeros((4, 100))


def main(args):
    makedirs("./figures", exist_ok=True)

    DEFAULT.musdb = args.musdb
    if args.data is None:
        args.data = DEFAULT.data

    args.device = "cpu" if args.cpu else "cuda"

    with torch.no_grad():
        COMMANDS[args.command](args)


COMMANDS = {
    "cross-likelihood": make_cross_likelihood_plot,
    "separate": make_separation_examples,
    "noise": make_noise_likelihood_plot,
    "posterior": make_posterior_examples,
    "toy-data": make_toy_dataset,
    "dist": make_data_distribution,
    "musdb-pre-save": make_musdb_pre_save,
}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", choices=COMMANDS.keys())
    parser.add_argument("--weights", type=get_newest_checkpoint)
    parser.add_argument("-k", choices=DEFAULT.all_signals)
    parser.add_argument("--data", type=path.abspath, default=None)
    parser.add_argument("-cpu", action="store_true")
    parser.add_argument("-musdb", action="store_true")
    main(parser.parse_args())
