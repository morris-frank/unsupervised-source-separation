import os
import time
from datetime import datetime
from statistics import mean
from typing import List, Callable

import torch
from torch import nn
from torch import optim
from torch.utils import data

from .log import log, MonkeyWriter


def train(model: nn.Module, loss_function: Callable, gpu: List[int],
          trainset: data.DataLoader, testset: data.DataLoader,
          iterations: int, use_board: bool = False):
    """
    :param model: The model to train
    :param loss_function: static loss function
    :param gpu: List of GPUs to use (int indexes)
    :param trainset: dataset for training data
    :param testset: dataset for testing data
    :param iterations: Number of iterations to run for
    :param use_board: Whether to use tensorboard
    :return:
    """
    model_id = f'{datetime.today():%y-%m-%d_%H}_{type(model).__name__}'

    # Setup logging and save stuff
    writer = MonkeyWriter()
    if use_board:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f'runs/{model_id}')

    os.makedirs('./checkpoints/', exist_ok=True)
    save_path = f'checkpoints/{model_id}_{{:06}}.pt'
    model_args = model.params

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
    train_iterator = iter(trainset)
    it_timer = time.time()
    for it in range(iterations):
        it_start_time = time.time()
        # Load next random batch
        try:
            x, y = next(train_iterator)
        except StopIteration:
            train_iterator = iter(trainset)
            x, y = next(train_iterator)

        model.train()
        loss = loss_function(model, x, y, device, it / iterations)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(it)

        losses.append(loss.detach().item())
        it_times.append(time.time() - it_start_time)

        # LOG INFO
        if it % 10 == 0 or it == iterations - 1:
            log(writer, it, {'Loss/train': mean(losses),
                             'Time/train': mean(it_times),
                             'LR': optimizer.param_groups[0]['lr']})
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
            test_time, test_losses = time.time(), []

            model.eval()
            ii = 0
            for x, y in testset:
                print(ii)
                ii += 1
                loss = loss_function(model, x, y, device, it / iterations)
                test_losses.append(loss.detach().item())

            log(writer, it, {'Loss/test': mean(test_losses),
                             'Time/test': time.time() - test_time})
