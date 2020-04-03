#!/usr/bin/env python
import os
from argparse import ArgumentParser

import torch
from torch import autograd

from thesis.data.toy import ToyData
from thesis.io import load_model, get_newest_file
from thesis.setup import TOY_SIGNALS, DEFAULT_DATA, IS_HERMES
from thesis.train import train


def _load_prior_networks(prefix: str = "", device="cuda"):
    return [
        load_model(get_newest_file("./checkpoints", f"{prefix}*Flowavenet*{s}*pt"), device).to(
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

    train_set = ToyData(path % "train", source=k, mel_source=True)
    test_set = ToyData(path % "test", source=k, mel_source=True)
    return model, train_set, test_set


def train_umix(path: str):
    from thesis.nn.models.umix import UMixer

    model = UMixer(width=128)
    model.name = "semi-supervised-fix-ampl"
    model.p_s = _load_prior_networks("Mar26")

    train_set = ToyData(
        path % "train", mix=True, mel=True, source=True, rand_amplitude=0.1
    )
    test_set = ToyData(
        path % "test", mix=True, mel=True, source=True, rand_amplitude=0.1
    )

    return model, train_set, test_set


def train_wn(path, signal):
    from thesis.nn.models.wn import WN
    k = TOY_SIGNALS.index(signal)

    model = WN(width=128)
    model.name = signal + "_Beta_denoise"
    model.p_s = [load_model(
        get_newest_file("./checkpoints", f"*Flowavenet*{signal}*pt"), "cuda"
    ).to("cuda")]

    train_set = ToyData(path % "train", source=k, rand_amplitude=0.1)
    test_set = ToyData(path % "test", source=k, rand_amplitude=0.1)
    return model, train_set, test_set


def train_nwn(path, signal):
    from thesis.nn.models.nwn import NWN
    k = TOY_SIGNALS.index(signal)

    model = NWN(width=128)
    model.name = signal + "_non_prob_denoise"
    model.p_s = [load_model(
        get_newest_file("./checkpoints", f"*Flowavenet*{signal}*pt"), "cuda"
    ).to("cuda")]

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
    "umix": train_umix,
    "wn": train_wn,
    "nwn": train_nwn,
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
    parser.add_argument("--iterations", default=50000, type=int)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("-debug", action="store_true")
    main(parser.parse_args())
