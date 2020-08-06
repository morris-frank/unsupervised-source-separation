#!/usr/bin/env python
from argparse import ArgumentParser
from itertools import product
from os import path, makedirs, getpid

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from colorama import Fore
from tqdm import tqdm, trange

from thesis import plot
from thesis.data.musdb import MusDBSamples
from thesis.data.toy import ToyData, generate_toy
from thesis.io import load_model, save_append, get_newest_checkpoint, appendz, \
    log_call
from thesis.setup import DEFAULT

mpl.use("agg")


@log_call(1)
def make_sample_from_prior(args, model=None):
    if model is None:
        model = load_model(args.weights, args.device)
    length = 8_000
    zshape = (1, 4, length)

    z = torch.randn(zshape, device=args.device)
    # z = torch.randn(1) * torch.ones(zshape)
    x = model.reverse(z)
    x = x[0, :, 3000:5000]
    x = x.clamp(-1.5, 1.5)
    x = x.cpu().numpy()

    appendz(args.results_file, samples=[x])


@log_call(1)
def make_const_logp(args, model=None):
    if model is None:
        model = load_model(args.weights, args.device)
    const_levels = np.linspace(-1, 1, 11)
    results = np.zeros((len(const_levels), 4))

    for i, level in enumerate(const_levels):
        x = level * torch.ones((1, 4, 8_000), device=args.device)
        log_p = model(x)[1][0, ...].mean(-1)
        results[i] = log_p.cpu().numpy()

    appendz(args.results_file, const_levels=const_levels, const_logp=results)


@log_call(1)
def make_noise_logp(args, model=None):
    if model is None:
        model = load_model(args.weights, args.device)
    noise_levels = np.linspace(-1, 1, 5)
    results = np.zeros((len(noise_levels), 4))

    for i, level in enumerate(noise_levels):
        x = level * torch.randn((1, 4, 8_000), device=args.device)
        log_p = model(x)[1][0, ...].mean(-1)
        results[i] = log_p.cpu().numpy()

    appendz(args.results_file, noise_levels=noise_levels, noise_logp=results)


@log_call(1)
def make_rel_noised_logp(args, model=None):
    if model is None:
        model = load_model(args.weights, args.device)
    batch_size = 5
    data = ToyData(
        args.data, "test", source=True, mel_source=False, length=4000
    ).loader(batch_size, drop_last=True)
    noise_levels = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    results = np.zeros((len(noise_levels), 4, len(data) * batch_size))
    for σ, (i, s) in product(noise_levels, enumerate(tqdm(data, leave=False))):
        s = s + σ * torch.randn_like(s)
        log_p = model(s.to(args.device))[1].mean(-1)
        results[σ][:, i * batch_size : (i + 1) * batch_size] = log_p.T.cpu().numpy()

    appendz(args.results_file, noised=results)


@log_call(1)
def make_rel_source_logp(args, model=None):
    if model is None:
        model = load_model(args.weights, args.device)

    N = 20
    if args.musdb:
        data = MusDBSamples(args.data, "test", space="time", length=16_384).loader(
            N, drop_last=True
        )
    else:
        data = ToyData(args.data, "test", source=True, length=4000).loader(
            N, drop_last=True
        )

    results = np.zeros((4, 4, len(data) * N))

    for i, s in enumerate(tqdm(data, leave=False)):
        *_, L = s.shape
        s = s.view(N * 4, 1, L).repeat(1, 4, 1).to(args.device)
        log_p = model(s)[1].mean(-1)
        results[:, :, (i * N) : ((i + 1) * N)] = (
            log_p.view(N, 4, 4).permute(1, 2, 0).squeeze().cpu().numpy()
        )
    appendz(args.results_file, channels=results)


def make_test_discrprior(args):
    from thesis.data.musdb import MusDBSamples2
    from thesis.nn.models.flowavenet import FlowavenetClassified

    batch_size = 24
    weights = get_newest_checkpoint("*FlowavenetClassified*")
    model = load_model(weights, args.device, model_class=FlowavenetClassified)
    test_set = MusDBSamples2(args.data, "test", complex=complex).loader(
        batch_size, drop_last=False
    )
    fp = f"./figures/{path.basename(weights).split('-')[0]}_prior_cross_entropy.npy"

    results = {"y": [], "ŷ": [], "logp": []}
    for k, (m, y) in enumerate(tqdm(test_set)):
        ŷ, logp, _ = model(m.to(args.device))
        results["y"].extend(y.squeeze().tolist())
        results["ŷ"].extend(ŷ.cpu().squeeze().tolist())
        results["logp"].extend(logp.cpu().squeeze().mean(-1).tolist())
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
    length, ns = 48_000, 4
    config = {"test": 500, "train": 5_000}

    for name, n in config.items():
        print(f"Generate Toy [{name}] n={n}, length={length}, ns={ns}")
        print(f"Save to {args.data}/{name}")
        makedirs(f"{args.data}/{name}/", exist_ok=True)
        for i in trange(n):
            item = generate_toy(length, ns)
            np.save(f"{args.data}/{name}/{name}_{i:05}.npy", item)


def make_musdb_dataset(args):
    from thesis.data.musdb import MusDB

    length, n = 48_000, 150

    for subset in ["test", "train"]:
        data = MusDB(args.data, subsets=subset, mel=True)
        fp = path.normpath(data.path) + "_samples/" + subset
        makedirs(fp, exist_ok=True)
        pid = getpid()
        for i, (wav, mel) in enumerate(
            tqdm(data.pre_save(n_per_song=n, length=length), total=len(data) * n)
        ):
            np.save(f"{fp}/{i // n:03}_{i % n:03}_{pid}_mel.npy", mel.numpy())
            np.save(f"{fp}/{i // n:03}_{i % n:03}_{pid}_time.npy", wav.numpy())


def make_langevin(args):
    from thesis.langevin import langevin_sample

    noise, length = 0.0, 16_384 // 4

    model = load_model(args.weights, args.device)
    σ = 0.1

    opt = (
        {"source": True, "mix": True}
        if "time" in model.name
        else {"mel_source": True, "mel_mix": True}
    )
    data = ToyData(
        args.data, "test", noise=noise, rand_amplitude=0.05, length=length, **opt,
    ).loader(1, shuffle=False)

    for i, (m, s) in enumerate(data):
        fig = plot.toy.reconstruction(s, sharey=True, ylim=[-1, 1])
        plt.savefig(f"s_{i:03}.png")
        plt.close(fig)
        for j, (ŝ, ℒ, δŝ) in enumerate(
            langevin_sample(model, σ, m.to(args.device), ŝ=s.clone().to(args.device))
        ):
            print(ℒ)
            δŝ /= δŝ.abs().max()
            fig = plot.toy.reconstruction(s, ŝ, δŝ, sharey=True, ylim=[-1, 1])
            plt.suptitle(ℒ)
            plt.savefig(f"ŝ_{i:03}_{j:04}.png")
            plt.close(fig)


def evaluate_prior(args):
    print(f"{Fore.YELLOW}With {Fore.GREEN}{args.basename}{Fore.RESET}:")
    model = load_model(args.weights, args.device)
    make_noise_logp(args, model=model)
    make_rel_source_logp(args, model=model)
    make_rel_noised_logp(args, model=model)
    for _ in range(10):
        make_sample_from_prior(args, model=model)


def main(args):
    makedirs("./figures", exist_ok=True)

    args.basename = path.basename(args.weights)[:-10]
    args.result_file = f"./figures/{args.basename}.npz"

    if args.command == "musdb":
        args.musdb = True

    DEFAULT.musdb = args.musdb
    if args.data is None:
        args.data = DEFAULT.data

    args.device = "cpu" if args.cpu else "cuda"

    with torch.no_grad():
        COMMANDS[args.command](args)


COMMANDS = {
    "channels": make_rel_source_logp,
    "separate": make_separation_examples,
    "noised": make_rel_noised_logp,
    "posterior": make_posterior_examples,
    "toy": make_toy_dataset,
    "musdb": make_musdb_dataset,
    "discrprior": make_test_discrprior,
    "langevin": make_langevin,
    "eval": evaluate_prior,
    "noise": make_noise_logp,
    "const": make_const_logp,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", choices=COMMANDS.keys())
    parser.add_argument("--weights", type=get_newest_checkpoint)
    parser.add_argument("-k", type=str)
    parser.add_argument("--data", type=path.abspath, default=None)
    parser.add_argument("-cpu", action="store_true")
    parser.add_argument("-musdb", action="store_true")
    main(parser.parse_args())
