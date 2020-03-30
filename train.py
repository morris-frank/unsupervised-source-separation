#!/usr/bin/env python
from argparse import ArgumentParser
from functools import partial
from os import path

import torch
from torch import autograd

from thesis.data.toy import ToyDataSourceK, ToyDataRandomAmplitude
from thesis.io import load_model, get_newest_file
from thesis.setup import TOY_SIGNALS, DEFAULT_DATA, IS_HERMES
from thesis.train import train


def _load_prior_networks(prefix: str = "", device="cuda"):
    return [
        load_model(get_newest_file("./checkpoints", f"{prefix}*{s}*pt"), device).to(
            device
        )
        for s in TOY_SIGNALS
    ]


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
    model.p_s = _load_prior_networks("Mar26")

    train_set = ToyDataRandomAmplitude(path=path % "train", min=0.9)
    test_set = ToyDataRandomAmplitude(path=path % "test", min=0.9)
    return model, train_set, test_set


def train_wn(path):
    from thesis.nn.models.wn import WN

    model = WN(width=128)
    model.p_s = load_model(get_newest_file("./checkpoints", f"*saw*pt"), "cuda").to(
        "cuda"
    )

    train_set = ToyDataRandomAmplitude(path=path % "train", min=0.9)
    test_set = ToyDataRandomAmplitude(path=path % "test", min=0.9)
    return model, train_set, test_set


def main(args):
    if IS_HERMES:
        args.batch_size = 2

    model, train_set, test_set = EXPERIMENTS[args.experiment](f"{args.data}/%s/")

    train_loader = train_set.loader(args.batch_size)
    test_loader = test_set.loader(args.batch_size)

    if args.debug:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    with autograd.set_detect_anomaly(args.debug):
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
    "wn": train_wn,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment", choices=EXPERIMENTS.keys())
    parser.add_argument(
        "--gpu", type=int, required=False, nargs="+",
    )
    parser.add_argument("--data", type=path.abspath, default=DEFAULT_DATA)
    parser.add_argument("-wandb", action="store_true", help="Logs to WandB.")
    parser.add_argument("--iterations", default=50000, type=int)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("-debug", action="store_true")
    main(parser.parse_args())
