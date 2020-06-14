#!/usr/bin/env python
import os
from argparse import ArgumentParser
from functools import partial

import torch
from torch import autograd

from thesis.data.toy import ToyData
from thesis.io import load_model, get_newest_checkpoint
from thesis.nn.models.denoiser import Denoiser, Denoiser_Semi
from thesis.setup import IS_HERMES, DEFAULT
from thesis.train import train


def train_prior(args):
    from thesis.nn.models.flowavenet import Flowavenet

    complex = False
    add = True

    _n = "_musdb" if args.musdb else "_toy"

    if args.signal is None:
        name = "all" + _n
        source = True
        groups = len(DEFAULT.signals)
    else:
        name = args.signal + _n
        k = DEFAULT.signals(args.signal)
        source = k
        groups = 1

    model = Flowavenet(
        in_channel=130 if complex else 80,
        n_block=8 if args.musdb else 4,
        n_flow=10 if args.musdb else 10,
        n_layer=4 if args.musdb else 4,
        block_per_split=2 if args.musdb else 2,
        width=48 if args.musdb else 32,
        name=name,
        groups=groups,
    )

    if args.musdb:
        from thesis.data.musdb import MusDBSamples

        train_set = MusDBSamples(args.data, "train", complex=complex)
        test_set = MusDBSamples(args.data, "test", complex=complex)
    else:
        set_opt = dict(
            source=source,
            mel_source=True,
            noise=0.03,
            rand_noise=True,
            interpolate=True,
            complex=complex,
            with_phase=add,
        )
        train_set = ToyData(args.data, "train", **set_opt)
        test_set = ToyData(args.data, "test", **set_opt)
    return model, train_set, test_set

def train_discprior(args):
    from thesis.nn.models.flowavenet import FlowavenetClassified

    name = "musdb" if args.musdb else "toy"

    model = FlowavenetClassified(
        in_channel=80,
        n_block=8 if args.musdb else 4,
        n_flow=10 if args.musdb else 10,
        n_layer=4 if args.musdb else 4,
        block_per_split=2 if args.musdb else 2,
        width=48 if args.musdb else 32,
        name=name,
    )

    if args.musdb:
        from thesis.data.musdb import MusDBSamples2

        train_set = MusDBSamples2(args.data, "train", complex=complex)
        test_set = MusDBSamples2(args.data, "test", complex=complex)
    return model, train_set, test_set


def train_jem(args):
    from thesis.nn.models.jem import JEM
    from thesis.data.musdb import MusDBSamples2

    groups = len(DEFAULT.signals)

    model = JEM(in_channels=80, out_channels=groups, width=48)

    train_set = MusDBSamples2(args.data, "train")
    test_set = MusDBSamples2(args.data, "test")
    return model, train_set, test_set


def train_demixer(path: str):
    from thesis.nn.models.demixer import Demixer

    model = Demixer(width=128, name="annil")
    model.p_s = [
        load_model(get_newest_checkpoint("*Discrim*"), device="cuda").to("cuda")
    ]

    set_opt = dict(mix=True, mel=True, source=True, rand_amplitude=0.1)
    train_set = ToyData(path, "train", **set_opt)
    test_set = ToyData(path, "test", **set_opt)
    return model, train_set, test_set


def train_denoiser(path: str, signal: str, modelclass):
    k = DEFAULT.signals.index(signal)

    model = modelclass(width=128, name=signal)
    model.p_s = [
        load_model(get_newest_checkpoint(f"*Flowavenet*{signal}"), "cuda").to("cuda")
    ]

    train_set = ToyData(path, "train", source=k, rand_amplitude=0.1)
    test_set = ToyData(path, "test", source=k, rand_amplitude=0.1)
    return model, train_set, test_set


def main(args):
    if IS_HERMES:
        args.batch_size = 2

    DEFAULT.musdb = args.musdb
    if args.data is None:
        args.data = DEFAULT.data

    model, train_set, test_set = EXPERIMENTS[args.experiment](args)

    print(f"pid is: {os.getpid()}")
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
    "jem": train_jem,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment", choices=EXPERIMENTS.keys())
    parser.add_argument("-k", choices=DEFAULT.all_signals, dest="signal")
    parser.add_argument(
        "--gpu", type=int, required=False, nargs="+",
    )
    parser.add_argument("--data", type=os.path.abspath, default=None)
    parser.add_argument("-wandb", action="store_true", help="Logs to WandB.")
    parser.add_argument("--iterations", default=50_000, type=int)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("-debug", action="store_true")
    parser.add_argument("-musdb", action="store_true")
    main(parser.parse_args())
