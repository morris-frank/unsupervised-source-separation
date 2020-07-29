import os
import time
from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Dict, List, Optional

import torch
from colorama import Fore
from torch import optim
from torch.nn.utils import clip_grad_value_
from torch.utils import data

from functools import reduce
from operator import add
from .io import glob_remove
from .nn.models import BaseModel
from .utils import max_grad, any_invalid_grad

LAST_LOG = defaultdict(float)
LAST_LOG["start"] = True

_wandb = None


def run_test_with_batch(model, batch, device):
    if isinstance(batch, list):
        if isinstance(batch[0], list) or isinstance(batch[0], tuple):
            (x1, x2), y = (
                (batch[0][0].to(device), batch[0][1].to(device)),
                batch[1].to(device),
            )
            ℒ = model.test((x1, x2), y)
        else:
            x, y = batch[0].to(device), batch[1].to(device)
            ℒ = model.test(x, y)
    else:
        x = batch.to(device)
        ℒ = model.test(x)
    return ℒ


def print_log(model: BaseModel, add_log: Dict, cat: str, step: Optional[int] = None):
    log = add_log.copy()

    # Add new logs from ℒ logger
    if hasattr(model, "L"):
        for k, v in model.ℒ.log.items():
            if len(v):
                log[f"{k}/{cat}"] = reduce(add, v) / len(v)
                model.ℒ.log[k] = []

    # Print to console
    _step = step if step is not None else "---"
    print(f"step {_step:>9} {Fore.YELLOW}[{cat}]", end=" ")
    for k, v in log.items():
        print(f"{Fore.RESET}{'/'.join(k.split('/')[:-1])}=", end="")
        col = (
            Fore.CYAN
            if v == LAST_LOG[k] or LAST_LOG["start"]
            else (Fore.GREEN if v < LAST_LOG[k] else Fore.RED)
        )
        print(f"{col}{v:.3e}", end=f"{Fore.RESET}, ")
        LAST_LOG[k] = v
    print()
    LAST_LOG["start"] = False

    if _wandb is not None:
        _wandb.log(log, step=step)


def test(model: BaseModel, test_loader: data.DataLoader, device: str):
    test_time, test_losses = time.time(), []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            ℒ = run_test_with_batch(model, batch, device)
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
    keep_checkpoints: bool = False,
    keep_optim: bool = False,
    base_lr: float = 1e-4,
    start_it: int = 0,
    optimizer_state_dict = None,
    scheduler_state_dict = None,
):
    """
    Args:
        model: the module to train
        gpu: list of GPUs to use (int indexes)
        train_loader: dataset loader for the training data
        test_loader: dataset loader for the test data
        iterations: number of iterations to train for
        wandb: Whether to log wandb
        keep_checkpoints: whether to keep all checkpoints not just the last one
        keep_optim: whether to also save the optimizer
        base_lr: the starting learing rate
    """
    model_id = f"{datetime.today():%b%d-%H%M}_{type(model).__name__}_{model.name}"

    os.makedirs("./checkpoints/", exist_ok=True)

    # Move model to device(s):
    device = f"cuda:{gpu[0]}" if gpu else "cpu"
    if gpu:
        model = model.to(device)
        # model = nn.DataParallel(model, device_ids=gpu)

    if wandb:
        global _wandb
        import wandb as __wandb

        _wandb = __wandb
        _wandb.init(
            name=model_id,
            config=model.params["kwargs"],
            project=__name__.split(".")[0],
        )

    # Setup optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), eps=1e-8, lr=base_lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    lr_milestones = torch.linspace(iterations * 0.36, iterations, 5).tolist()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_milestones, gamma=0.6)
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    losses, it_times = [], []
    train_iterator = iter(train_loader)
    it_timer = time.time()
    model.train()
    print(
        f"\n{Fore.YELLOW}This is {Fore.GREEN}{model_id}{Fore.RESET}\n"
        f"{Fore.YELLOW}{f'{Fore.GREEN} Start training {Fore.YELLOW}'.center(80, '-')}{Fore.RESET}"
    )
    for it in range(start_it, iterations):
        it_start_time = time.time()
        # Load next random batch
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        ℒ = run_test_with_batch(model, batch, device)
        model.zero_grad()

        if torch.isnan(ℒ) or torch.isinf(ℒ):
            print(
                Fore.RED + "NaN Loss ℒ.\n"
                "Try Again. I'm gonna try to continue…" + Fore.RESET
            )
            exit()
        else:
            ℒ.backward()
            clip_grad_value_(model.parameters(), 30)
            if any_invalid_grad(model.parameters()):
                print(
                    Fore.RED + "There was a NaN or inf in one of the grads.\n"
                    "Saving everything……" + Fore.RESET
                )
                save_point = {
                    "model_state_dict": model.state_dict(),
                    "params": model.params,
                    "batch": batch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "it": it,
                    "scheduler": scheduler.state_dict(),
                }
                torch.save(
                    save_point, f"checkpoints/invalid_grad_{model_id}_{it:06}.pt"
                )
                exit()
            optimizer.step()
            scheduler.step()

        losses.append(ℒ.detach().item())
        it_times.append(time.time() - it_start_time)

        # LOG INFO (every 10 mini batches)
        if it % 10 == 0 or it == iterations - 1:
            log = {
                "Loss/train": mean(losses),
                "Time/train": mean(it_times),
                "LR/train": optimizer.param_groups[0]["lr"],
                "MaxGrad/train": max_grad(model.parameters()),
            }
            print_log(model, log, "train", step=it)
            losses, it_times = [], []

        # TEST AND SAVE THE MODEL (every 30min)
        if (time.time() - it_timer) > 1800 or it == iterations - 1:
            if not keep_checkpoints:
                glob_remove(f"checkpoints/{model_id}_*.pt")
            save_point = {
                "it": it,
                "model_state_dict": model.state_dict(),
                "params": model.params,
            }
            if keep_optim:
                save_point.update(
                    {"optimizer_state_dict": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(),
                     "test": ℒ}
                )
            torch.save(save_point, f"checkpoints/{model_id}_{it:06}.pt")
            test(model, test_loader, device)
            it_timer = time.time()
            model.train()
