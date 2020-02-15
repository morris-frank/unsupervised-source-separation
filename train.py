#!/usr/bin/env python
from argparse import ArgumentParser
from os import path, uname

from torch import autograd

from thesis.data.wrapper import map_dataset
from thesis.train import train
from thesis.utils import optional


def glow():
    from thesis.nn.models.waveglow import WaveGlow
    max_batch_size = 32
    model = WaveGlow(channels=4, n_flows=10, wn_layers=12)  # rf: 2^11
    return model, max_batch_size


def one_channel_conditioned():
    from thesis.nn.models.nvp import ConditionalRealNVP

    max_batch_size = 26
    model = ConditionalRealNVP(
        classes=4, n_flows=15, wn_layers=10, wn_width=64
    )  # rf: 2*2**9
    return model, max_batch_size


def hydra():
    from thesis.nn.models.hydra import Hydra

    max_batch_size = 18
    model = Hydra(classes=4, in_channels=1, out_channels=101, wn_width=32)
    return model, max_batch_size


def monet():
    from thesis.nn.models.monet import MONet

    max_batch_size = 75
    model = MONet(slots=4)
    return model, max_batch_size


def main(args):
    if args.experiment not in EXPERIMENTS:
        raise ValueError("Invalid experiment given.")

    model, max_batch_size = EXPERIMENTS[args.experiment]()

    batch_size = args.batch_size or max_batch_size
    if uname().nodename == 'hermes':
        batch_size = 2
    train_loader = map_dataset(model, args.data, "train").loader(batch_size)
    test_loader = map_dataset(model, args.data, "test").loader(batch_size)

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
    "glow": glow,
    "1cc": one_channel_conditioned,
    "one_channel_conditioned": one_channel_conditioned,
    "hydra": hydra,
    "monet": monet,
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
        type=path.abspath,
        required=True,
        help="The top-level directory of dataset.",
    )
    parser.add_argument("-wandb", action="store_true", help="Logs to WandB.")
    parser.add_argument("--iterations", default=50000, type=int)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("-debug", action="store_true")
    main(parser.parse_args())
