#!/usr/bin/env python
import os
from argparse import ArgumentParser
from functools import partial

from colorama import Fore
from torch import autograd

from thesis.data.toy import ToyDataSourceK, ToyData, ToyDataRandomAmplitude, TOY_SIGNALS
from thesis.io import load_model, get_newest_file
from thesis.train import train
from thesis.utils import optional


def _load_prior_networks(prefix: str = "", device="cuda"):
    priors = []
    for source in TOY_SIGNALS:
        weight = get_newest_file("./checkpoints", f"{prefix}*{source}*pt")
        print(
            f"{Fore.YELLOW}For {Fore.GREEN}{source} {Fore.YELLOW}we using\t{Fore.GREEN}{weight}{Fore.RESET}"
        )
        priors.append(load_model(weight, device).to(device))
    return priors


def train_prior(path: str, signal: str):
    from thesis.nn.models.flowavenet import Flowavenet

    k = TOY_SIGNALS.index(signal)

    mel_channels = 80
    model = Flowavenet(
        in_channel=1,
        cin_channel=mel_channels,
        n_block=4,
        n_flow=10,
        n_layer=4,
        block_per_split=2,
        width=48,
        name=signal,
    )
    train_set = ToyDataSourceK(path=path % "train", k=k, mel=True)
    test_set = ToyDataSourceK(path=path % "test", k=k, mel=True)
    return model, train_set, test_set


def train_umix(path: str):
    from thesis.nn.models.umix import UMixer

    model = UMixer(width=128)
    model.name = "semi-supervised-fix-ampl"
    model.p_s = _load_prior_networks("Mar22")

    train_set = ToyData(path=path % "train", mel=True, sources=True)
    test_set = ToyData(path=path % "test", mel=True, sources=True)
    return model, train_set, test_set


def train_cumix(path: str):
    from thesis.nn.models.cumix import CUMixer

    model = CUMixer(mu=101, width=128)

    train_set = ToyDataRandomAmplitude(path=path % "train")
    test_set = ToyDataRandomAmplitude(path=path % "test")
    return model, train_set, test_set


def train_numix(path: str):
    from thesis.nn.models.numix import NUMixer

    model = NUMixer(width=128)

    train_set = ToyDataRandomAmplitude(path=path % "train")
    test_set = ToyDataRandomAmplitude(path=path % "test")
    return model, train_set, test_set


def main(args):
    if args.experiment not in EXPERIMENTS:
        raise ValueError("Invalid experiment given.")

    model, train_set, test_set = EXPERIMENTS[args.experiment](f"{args.data}/%s/")

    if os.uname().nodename == "hermes":
        args.batch_size = 2
    train_loader = train_set.loader(args.batch_size)
    test_loader = test_set.loader(args.batch_size)

    with optional(args.debug, autograd.detect_anomaly()):
        train(
            model=model,
            gpu=args.gpu,
            train_loader=train_loader,
            test_loader=test_loader,
            iterations=args.iterations,
            wandb=args.wandb,
        )


EXPERIMENTS = {
    "sin": partial(train_prior, signal="sin"),
    "square": partial(train_prior, signal="square"),
    "saw": partial(train_prior, signal="saw"),
    "triangle": partial(train_prior, signal="triangle"),
    "umix": train_umix,
    "numix": train_numix,
    "cumix": train_cumix,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment", type=str, help="choose the experiment")
    parser.add_argument(
        "--gpu",
        type=int,
        required=False,
        nargs="+",
        help="The GPU ids to use. If unset, will use CPU.",
    )
    parser.add_argument(
        "--data",
        type=os.path.abspath,
        required=True,
        help="The top-level directory of dataset.",
    )
    parser.add_argument("-wandb", action="store_true", help="Logs to WandB.")
    parser.add_argument("--iterations", default=50000, type=int)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("-debug", action="store_true")
    main(parser.parse_args())
