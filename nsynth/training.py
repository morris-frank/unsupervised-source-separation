import os
import time
from datetime import datetime
from statistics import mean
from typing import List, Dict, Callable

import torch
from torch import nn
from torch import optim
from torch.utils import data

from .logging import ConfusionMatrix, log, MonkeyWriter
from .modules import AutoEncoder


def train(model: AutoEncoder, loss_function: Callable, gpu: List[int],
          trainset: data.DataLoader, testset: data.DataLoader,
          num_iter: int, use_board: bool = False,
          save_suffix: str = '', iterpoints: Dict = None):
    """
    :param model: The WaveNet model Module
    :param loss_function: The static loss function, should take params:
        (model: Module, x: Tensor, y: Tensor, device: str)
    :param gpu: List of GPUs to use (int indexes)
    :param trainset: The dataset for training data
    :param testset: the dataset for testing data
    :param iterpoints: The number of iterations to print, save and test
    :param num_iter: Number of iterations
    :param use_board: Whether to use tensorboard
    :param save_suffix:
    :return:
    """
    model_id = f'{datetime.today():%y-%m-%d_%H}_{type(model).__name__}_' \
               f'{save_suffix}'
    # Setup logging and save stuff
    writer = MonkeyWriter()
    if use_board:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f'runs/{model_id}')

    os.makedirs('./models/', exist_ok=True)
    save_path = f'models/{model_id}_{{:06}}.pt'
    model_args = model.params

    if iterpoints is None:
        iterpoints = {'print': 20, 'save': 5000, 'test': 500}

    # Move model to device(s):
    device = f'cuda:{gpu[0]}' if gpu else 'cpu'
    if gpu:
        model = nn.DataParallel(model.to(device), device_ids=gpu)

    # Setup optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), eps=1e-8, lr=1e-3)
    lr_milestones = torch.linspace(num_iter * 0.36, num_iter, 5).tolist()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_milestones,
                                               gamma=0.6)

    losses, it_times = [], []
    iloader = iter(trainset)
    for it in range(num_iter):
        it_start_time = time.time()
        # Load next random batch
        try:
            x, y = next(iloader)
        except StopIteration:
            iloader = iter(trainset)
            x, y = next(iloader)

        model.train()
        loss, _ = loss_function(model, x, y, device, it / num_iter)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(it)

        losses.append(loss.detach().item())
        it_times.append(time.time() - it_start_time)

        # LOG INFO
        if it % iterpoints['print'] == 0 or it == num_iter - 1:
            log(writer, it, {'Loss/train': mean(losses),
                             'Time/train': mean(it_times),
                             'LR': optimizer.param_groups[0]['lr']})
            losses, it_times = [], []

        # SAVE THE MODEL
        if it % iterpoints['save'] == 0 or it == num_iter - 1:
            torch.save({
                'it': it,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'params': model_args,
            }, save_path.format(it))

        # TEST THE MODEL
        if it % iterpoints['test'] == 0 or it == num_iter - 1:
            test_time, test_losses = time.time(), []
            conf_mat = ConfusionMatrix()

            model.eval()
            ii = 0
            for x, y in testset:
                print(ii)
                ii += 1
                loss, y_ = loss_function(model, x, y, device, it / num_iter)
                test_losses.append(loss.detach().item())
                if y_:
                    conf_mat.add(y_, y)

            log(writer, it, {'Loss/test': mean(test_losses),
                             'Class confusion': conf_mat.plot(),
                             'Time/test': time.time() - test_time})

    print(f'FINISH {num_iter} mini-batches')
