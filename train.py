#!/usr/bin/env python
from argparse import ArgumentParser
from functools import partial
import os

from torch import autograd

from thesis.train import train
from thesis.utils import optional
from thesis.data.toy import ToyDataSourceK


def prior_flo(path: str, k: int):
    from thesis.nn.models.flowavenet import Flowavenet

    mel_channels = 80
    model = Flowavenet(
        in_channel=1,
        cin_channel=mel_channels,
        n_block=6,
        n_flow=4,
        n_layer=2,
        affine=True,
        block_per_split=3,
        name=k
    )

    train_set = ToyDataSourceK(path=path % "train", k=k, mel=True)
    test_set = ToyDataSourceK(path=path % "test", k=k, mel=True)
    return model, train_set, test_set


def main(args):
    if args.experiment not in EXPERIMENTS:
        raise ValueError("Invalid experiment given.")

    model, train_set, test_set = EXPERIMENTS[args.experiment](f"{args.data}/{{}}/")

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
    "prior-0": partial(prior_flo, k=0),
    "prior-1": partial(prior_flo, k=1),
    "prior-2": partial(prior_flo, k=2),
    "prior-3": partial(prior_flo, k=3),
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
