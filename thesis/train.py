import os
import time
from datetime import datetime
from statistics import mean
from typing import Dict
from typing import List, Callable

import torch
import wandb as _wandb
from torch import nn
from torch import optim
from torch.utils import data


def _print_log(items: Dict, step: int):
    print(f'step {step:>9}', end='\t')
    for k, v in items.items():
        print(f'{k}={v:.3e}, ', end='')
    print()


def _test(model: nn.Module, loss_function: Callable,
          test_loader: data.DataLoader, it: int, iterations: int, wandb: bool):
    test_time, test_losses = time.time(), []
    model.eval()
    for x, y in test_loader:
        loss = loss_function(model, x, y, it / iterations)
        test_losses.append(loss.detach().item())

    log = {'Loss/test': mean(test_losses),
           'Time/test': time.time() - test_time}

    _print_log(log, step=it)
    if wandb:
        _wandb.log(log, step=it)


def train(model: nn.Module, loss_function: Callable, gpu: List[int],
          train_loader: data.DataLoader, test_loader: data.DataLoader,
          iterations: int, wandb: bool = False):
    """
    :param model: The model to train
    :param loss_function: static loss function
    :param gpu: List of GPUs to use (int indexes)
    :param train_loader: dataset for training data
    :param test_loader: dataset for testing data
    :param iterations: Number of iterations to run for
    :param wandb: Whether to log to wandb
    :return:
    """
    model_id = f'{datetime.today():%y-%m-%d_%H}_{type(model).__name__}'

    os.makedirs('./checkpoints/', exist_ok=True)
    save_path = f'checkpoints/{model_id}_{{:06}}.pt'
    model_args = model.params

    if wandb:
        _wandb.init(name=model_id, config=model_args['kwargs'],
                    project=__name__.split('.')[0])

    # Move model to device(s):
    device = f'cuda:{gpu[0]}' if gpu else 'cpu'
    if gpu:
        model = nn.DataParallel(model.to(device), device_ids=gpu)

    # Setup optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), eps=1e-8, lr=1e-3)
    lr_milestones = torch.linspace(iterations * 0.36, iterations, 5).tolist()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_milestones,
                                               gamma=0.6)
    test_at = iterations // 10

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
        loss = loss_function(model, x, y, it / iterations)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(it)

        losses.append(loss.detach().item())
        it_times.append(time.time() - it_start_time)

        # LOG INFO (every 10 mini batches)
        if it % 10 == 0 or it == iterations - 1:
            log = {'Loss/train': mean(losses),
                   'Time/train': mean(it_times),
                   'LR': optimizer.param_groups[0]['lr']}
            _print_log(log, step=it)
            if wandb:
                _wandb.log(log, step=it)
            losses, it_times = [], []

        # SAVE THE MODEL (every 30min)
        if (time.time() - it_timer) > 1800 or it == iterations - 1:
            torch.save({
                'it': it,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'params': model_args,
            }, save_path.format(it))
            it_timer = time.time()

        # TEST THE MODEL
        if it % test_at == 0 or it == iterations - 1:
            _test(model, loss_function, test_loader, it, iterations, wandb)
