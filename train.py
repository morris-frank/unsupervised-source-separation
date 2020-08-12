#!/usr/bin/env python
import os
from argparse import ArgumentParser
from functools import partial

import torch
from torch import autograd

from thesis.data.toy import ToyData
from thesis.data.musdb import MusDBSamples
from thesis.io import load_model, get_newest_checkpoint
from thesis.nn.models.denoiser import Denoiser
from thesis.setup import IS_HERMES, DEFAULT
from thesis.train import train


def train_prior_time(args, rand_ampl=0.2, classified=False):
    from thesis.nn.models.flowavenet import Flowavenet, FlowavenetClassified
    modelclass = FlowavenetClassified if classified else Flowavenet

    if args.signal is None:
        name = "musdb" if args.musdb else "toy"
        source, groups = True, len(DEFAULT.signals)
        groups = len(DEFAULT.signals)
    else:
        name = args.signal
        k = DEFAULT.signals(args.signal)
        source, groups = k, 1

    name += f"_time"
    if args.noise is not None:
        name += f"_noise_{str(args.noise).replace('.', '-')}"
    else:
        args.noise = 0
    if rand_ampl > 0:
        name += "_rand_ampl"

    width = 48 if args.musdb else 32

    model = modelclass(
        in_channel=1,
        n_block=8 if args.musdb else 4,
        n_flow=6,
        n_layer=10,
        block_per_split=2,
        width=width,
        name=name,
        groups=groups,
    )

    if args.musdb:
        train_set = MusDBSamples(args.data, "train", space="time", length=args.length)
        test_set = MusDBSamples(args.data, "test", space="time", length=args.length)
    else:
        opt = dict(
            noise=args.noise, rand_amplitude=rand_ampl, length=args.length, source=source, shuffle_indexed=classified
        )
        train_set = ToyData(args.data, "train", **opt)
        test_set = ToyData(args.data, "test", **opt)
    return model, train_set, test_set


def train_prior_mel(args, noise=0.0):
    from thesis.nn.models.glow import Glow
    args.length = 240

    if args.signal is None:
        name = "musdb" if args.musdb else "toy"
        source, groups = True, len(DEFAULT.signals)
        groups = len(DEFAULT.signals)
    else:
        name = args.signal
        k = DEFAULT.signals(args.signal)
        source, groups = k, 1

    name += f"_mel"
    if noise > 0:
        name += "_noise"

    model = Glow(in_channel=4,
                 n_flow=32 if args.musdb else 16,
                 n_block=4, groups=groups, name=name)

    if args.musdb:
        train_set = MusDBSamples(args.data, "train", space="mel", length=args.length)
        test_set = MusDBSamples(args.data, "test", space="mel", length=args.length)
    else:
        opt = dict(
            noise=noise, length=args.length, mel_source=source
        )
        train_set = ToyData(args.data, "train", **opt)
        test_set = ToyData(args.data, "test", **opt)
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


def train_denoiser(args, modelclass):
    model = modelclass(width=128)

    model.p_s = [
        load_model(get_newest_checkpoint(f"{args.signal}*Flowavenet*"), "cuda").to(
            "cuda"
        )
    ]

    train_set = ToyData(args.data, "train", source=True, rand_amplitude=0.1)
    test_set = ToyData(args.data, "test", source=True, rand_amplitude=0.1)
    return model, train_set, test_set


def main(args):
    if IS_HERMES:
        args.batch_size = 2

    DEFAULT.musdb = args.musdb
    if args.data is None:
        args.data = DEFAULT.data

    model, train_set, test_set = EXPERIMENTS[args.experiment](args)
    optimizer_state_dict, scheduler_state_dict, start_it = None, None, 0

    if args.weights is not None:
        device = f"cuda:{args.gpu[0]}" if args.gpu else "cpu"
        spt = torch.load(get_newest_checkpoint(args.weights), map_location=torch.device(device))
        state = model.state_dict()
        state.update(spt['model_state_dict'])
        model.load_state_dict(state)
        # optimizer_state_dict = spt['optimizer_state_dict']
        scheduler_state_dict = spt['scheduler']
        start_it = spt["it"]
        for module in model.modules():
            if hasattr(module, "initialized"):
                module.initialized = True

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
            keep_optim=True,
            base_lr=args.base_lr,
            start_it=start_it,
            optimizer_state_dict=optimizer_state_dict,
            scheduler_state_dict=scheduler_state_dict,
        )


EXPERIMENTS = {
    "prior_time": train_prior_time,
    "prior_mel": train_prior_mel,
    "prior_time_noised": train_prior_time,
    "prior_mel_noised": train_prior_mel,
    "demixer": train_demixer,
    "denoiser": partial(train_denoiser, modelclass=Denoiser),
    "discrprior": partial(train_prior_time, classified=True),
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment", choices=EXPERIMENTS.keys())
    parser.add_argument("-k", type=str, dest="signal")
    parser.add_argument(
        "--gpu", type=int, required=False, nargs="+",
    )
    parser.add_argument("--data", type=os.path.abspath, default=None)
    parser.add_argument("-wandb", action="store_true", help="Logs to WandB.")
    parser.add_argument("--iterations", default=250_000, type=int)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("-debug", action="store_true")
    parser.add_argument("-musdb", action="store_true")
    parser.add_argument("-L", type=int, default=16_384, dest='length')
    parser.add_argument("-lr", type=float, default=1e-4, dest='base_lr')
    parser.add_argument("--weights", type=str)
    parser.add_argument("-noise", type=float)
    main(parser.parse_args())
