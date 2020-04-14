#!/usr/bin/env python
import os
from argparse import ArgumentParser
from functools import partial

import torch
from torch import autograd

from thesis.data.toy import ToyData, ToyDataAndNoise, RandToyData
from thesis.io import load_model, get_newest_checkpoint
from thesis.nn.models.denoiser import Denoiser, GAN, Denoiser_Semi
from thesis.setup import TOY_SIGNALS, DEFAULT_DATA, IS_HERMES
from thesis.train import train


def _load_prior_networks(prefix: str = "Apr06", device="cuda"):
    return [
        load_model(get_newest_checkpoint(f"{prefix}*Flowavenet*{s}"),
                   device).to(device)
        for s in TOY_SIGNALS
    ]


def train_prior(path: str, signal: str):
    from thesis.nn.models.flowavenet import Flowavenet

    k = TOY_SIGNALS.index(signal)

    model = Flowavenet(
        in_channel=1,
        cin_channel=80,
        n_block=4,
        n_flow=10,
        n_layer=4,
        block_per_split=2,
        width=48,
        name=signal,
    )

    train_set = ToyDataAndNoise(
        path % "train", source=k, mel_source=True, noise=0.03, rand_noise=True
    )
    test_set = ToyDataAndNoise(
        path % "test", source=k, mel_source=True, noise=0.03, rand_noise=True
    )
    return model, train_set, test_set


def train_discriminator(path: str):
    from thesis.nn.models.discriminator import Discriminator

    model = Discriminator(n_classes=4, width=48, mel_channels=80)

    train_set = RandToyData(path % "train", source=True, mel_source=True,
                            noise=0.03, rand_noise=True)
    test_set = RandToyData(path % "test", source=True, mel_source=True,
                           noise=0.03, rand_noise=True)

    return model, train_set, test_set


def train_demixer(path: str):
    from thesis.nn.models.demixer import Demixer

    model = Demixer(width=128, name="annil")
    # model.p_s = _load_prior_networks()
    model.p_s = [load_model(get_newest_checkpoint('*Discrim*'), device="cuda").to("cuda")]

    train_set = ToyData(
        path % "train", mix=True, mel=True, source=True, rand_amplitude=0.1
    )
    test_set = ToyData(
        path % "test", mix=True, mel=True, source=True, rand_amplitude=0.1
    )

    return model, train_set, test_set


def train_denoiser(path: str, signal: str, modelclass):
    k = TOY_SIGNALS.index(signal)

    model = modelclass(width=128, name=signal)
    model.p_s = [
        load_model(get_newest_checkpoint(f"*Flowavenet*{signal}"), "cuda").to(
            "cuda")
    ]

    train_set = ToyData(path % "train", source=k, rand_amplitude=0.1)
    test_set = ToyData(path % "test", source=k, rand_amplitude=0.1)
    return model, train_set, test_set


def main(args):
    if IS_HERMES:
        args.batch_size = 2

    fa = {"path": f"{args.data}/%s/"}
    if args.signal is not None:
        fa["signal"] = args.signal
    model, train_set, test_set = EXPERIMENTS[args.experiment](**fa)

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
    "prior": train_prior,
    "demixer": train_demixer,
    "denoiser": partial(train_denoiser, modelclass=Denoiser),
    "denoiser_semi": partial(train_denoiser, modelclass=Denoiser_Semi),
    "gan": partial(train_denoiser, modelclass=GAN),
    "discr": train_discriminator,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment", choices=EXPERIMENTS.keys())
    parser.add_argument("-k", choices=TOY_SIGNALS, dest="signal")
    parser.add_argument(
        "--gpu", type=int, required=False, nargs="+",
    )
    parser.add_argument("--data", type=os.path.abspath, default=DEFAULT_DATA)
    parser.add_argument("-wandb", action="store_true", help="Logs to WandB.")
    parser.add_argument("--iterations", default=21_000, type=int)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("-debug", action="store_true")
    main(parser.parse_args())
