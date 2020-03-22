#!/usr/bin/env python
import os
from argparse import ArgumentParser
from functools import partial

from torch import autograd

from thesis.data.toy import ToyDataSourceK, ToyDataMixes
from thesis.io import load_model
from thesis.train import train
from thesis.utils import optional, get_newest_file
from colorama import Fore

signals = ["sin", "square", "saw", "triangle"]


def train_prior(path: str, k: int):
    from thesis.nn.models.flowavenet import Flowavenet

    mel_channels = 80
    model = Flowavenet(
        in_channel=1,
        cin_channel=mel_channels,
        n_block=4,
        n_flow=6,
        n_layer=2,
        block_per_split=2,
        width=48,
        name=signals[k],
    )
    train_set = ToyDataSourceK(path=path % "train", k=k, mel=True)
    test_set = ToyDataSourceK(path=path % "test", k=k, mel=True)
    return model, train_set, test_set


def train_umix(path: str):
    from thesis.nn.models.umix import UMixer

    priors = []
    for source in ['sin', 'square', 'saw', 'triangle']:
        weight = get_newest_file("./checkpoints", f"*{source}*pt")
        print(f"{Fore.YELLOW}For {Fore.GREEN}{source} {Fore.YELLOW}we using {Fore.GREEN}{weight}{Fore.RESET}")
        priors.append(load_model(weight, "cuda").to("cuda"))

    model = UMixer(width=128)
    model.p_s = priors

    train_set = ToyDataMixes(path=path % "train", mel=True, sources=True)
    test_set = ToyDataMixes(path=path % "test", mel=True, sources=True)
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
    "prior-0": partial(train_prior, k=0),
    "prior-1": partial(train_prior, k=1),
    "prior-2": partial(train_prior, k=2),
    "prior-3": partial(train_prior, k=3),
    "umix": train_umix,
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
