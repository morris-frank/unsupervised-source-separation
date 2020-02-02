from argparse import ArgumentParser
from math import log2
from os import path

from thesis.data.toy import ToyData, ToyDataSingle
from thesis.nn.models import WaveGlow, MultiRealNVP, ConditionalRealNVP
from thesis.train import train


def four_channel_unconditioned(data):
    receptive_field = 2 ** 11
    batch_size = 12

    model = WaveGlow(channels=4, n_flows=15,
                     wn_layers=int(log2(receptive_field) + 1))
    loss_function = model.loss(σ=1.)

    train_loader = ToyData(f'{data}/toy_train.npy', receptive_field) \
        .loader(batch_size)
    test_loader = ToyData(f'{data}/toy_test.npy', receptive_field) \
        .loader(batch_size)
    return model, loss_function, train_loader, test_loader


def one_channel_unconditioned(data):
    receptive_field = 2 * 2 ** 9
    batch_size = 16

    model = MultiRealNVP(channels=4, n_flows=15,
                         wn_layers=int(log2(receptive_field // 2) + 1))
    loss_function = model.loss(σ=1.)

    train_loader = ToyData(f'{data}/toy_train.npy', receptive_field) \
        .loader(batch_size)
    test_loader = ToyData(f'{data}/toy_test.npy', receptive_field) \
        .loader(batch_size)
    return model, loss_function, train_loader, test_loader


def one_channel_conditioned(data):
    receptive_field = 2 * 2 ** 9
    batch_size = 26

    model = ConditionalRealNVP(classes=4, n_flows=15,
                               wn_layers=int(log2(receptive_field // 2) + 1),
                               wn_width=64)
    loss_function = model.loss()

    train_loader = ToyDataSingle(f'{data}/toy_train.npy', receptive_field) \
        .loader(batch_size)
    test_loader = ToyDataSingle(f'{data}/toy_test.npy', receptive_field) \
        .loader(batch_size)
    return model, loss_function, train_loader, test_loader


def main(args):
    if args.experiment == 'four_channel_unconditioned':
        func = four_channel_unconditioned
    elif args.experiment == 'one_channel_unconditioned':
        func = one_channel_unconditioned
    elif args.experiment == 'one_channel_conditioned':
        func = one_channel_conditioned
    else:
        raise ValueError('Invalid experiment given.')

    model, loss_function, train_loader, test_loader = func(args.data)
    train(model=model, loss_function=loss_function, gpu=args.gpu,
          train_loader=train_loader, test_loader=test_loader,
          iterations=args.iterations, wandb=args.wandb,
          skip_test=args.skip_test)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('experiment', type=str, help='choose the experiment')
    parser.add_argument('--gpu', type=int, required=False, nargs='+',
                        help='The GPU ids to use. If unset, will use CPU.')
    parser.add_argument('--data', type=path.abspath, required=True,
                        help='The top-level directory of dataset.')
    parser.add_argument('-wandb', action='store_true',
                        help='Logs to WandB.')
    parser.add_argument('--iterations', default=50000, type=int)
    parser.add_argument('-notest', action='store_true', dest='skip_test')
    main(parser.parse_args())
