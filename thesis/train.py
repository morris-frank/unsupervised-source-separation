from typing import Optional
import os
import time
from datetime import datetime
from statistics import mean
from typing import Dict
from typing import List

import torch
import wandb as _wandb
from torch import optim
from torch.utils import data
from torch.nn.utils import clip_grad_norm_
from colorama import Fore

from .nn.models import BaseModel
from collections import defaultdict


LAST_LOG = defaultdict(float)


def print_log(model: BaseModel, add_log: Dict, cat: str, step: Optional[int] = None):
    log = add_log.copy()

    # Add new logs from ℒ logger
    if hasattr(model, "L"):
        for k, v in model.ℒ.log.items():
            log[f"{k}/{cat}"] = mean(v)
            model.ℒ.log[k] = []

    # Print to console
    _step = step if step is not None else "---"
    print(f"step {_step:>9}", end="\t")
    for k, v in log.items():
        col = (
            Fore.CYAN
            if v == LAST_LOG[k]
            else (Fore.GREEN if v < LAST_LOG[k] else Fore.RED)
        )
        print(f"{Fore.RESET}{k}={col}{v:.3e}{Fore.RESET}, ", end="")
        LAST_LOG[k] = v
    print()

    if _wandb.run is not None:
        _wandb.log(log, step=step)


def test(model: BaseModel, test_loader: data.DataLoader, device: str):
    test_time, test_losses = time.time(), []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            ℒ = model.test(x, y)
            test_losses.append(ℒ.detach().item())

    log = {"Loss/test": mean(test_losses), "Time/test": time.time() - test_time}

    print_log(model, log, "test")


def train(
    model: BaseModel,
    gpu: List[int],
    train_loader: data.DataLoader,
    test_loader: data.DataLoader,
    iterations: int,
    wandb: bool = False,
):
    """
    Args:
        model: the module to train
        gpu: list of GPUs to use (int indexes)
        train_loader: dataset loader for the training data
        test_loader: dataset loader for the test data
        iterations: number of iterations to train for
        wandb: Whether to log wandb
    """
    model_id = f"{datetime.today():%b%d-%H%M}_{type(model).__name__}_{model.name}"

    os.makedirs("./checkpoints/", exist_ok=True)

    # Move model to device(s):
    device = f"cuda:{gpu[0]}" if gpu else "cpu"
    if gpu:
        model = model.to(device)
        # model = nn.DataParallel(model, device_ids=gpu)

    if wandb:
        _wandb.init(
            name=model_id,
            config=model.params["kwargs"],
            project=__name__.split(".")[0],
        )

    # Setup optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), eps=1e-8, lr=1e-3)
    lr_milestones = torch.linspace(iterations * 0.36, iterations, 5).tolist()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_milestones, gamma=0.6)

    losses, it_times = [], []
    train_iterator = iter(train_loader)
    it_timer = time.time()
    for it in range(iterations):
        it_start_time = time.time()
        # Load next random batch
        try:
            x, y = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            x, y = next(train_iterator)

        model.train()
        model.zero_grad()

        x, y = x.to(device), y.to(device)
        ℒ = model.test(x, y)

        if torch.isnan(ℒ):
            print(Fore.RED + 'nan loss. skip optim!' + Fore.RESET)
        else:
            ℒ.backward()
            clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            scheduler.step(it)

        losses.append(ℒ.detach().item())
        it_times.append(time.time() - it_start_time)

        # LOG INFO (every 10 mini batches)
        if it % 10 == 0 or it == iterations - 1:
            log = {
                "Loss/train": mean(losses),
                "Time/train": mean(it_times),
                "LR": optimizer.param_groups[0]["lr"],
            }
            print_log(model, log, "train", step=it)
            losses, it_times = [], []

        # TEST AND SAVE THE MODEL (every 30min)
        if (time.time() - it_timer) > 1800 or it == iterations - 1:
            torch.save(
                {
                    "it": it,
                    "model": model,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test": ℒ,
                    "params": model.params,
                },
                f"checkpoints/{model_id}_{it:06}.pt",
            )
            test(model, test_loader, device)
            it_timer = time.time()
