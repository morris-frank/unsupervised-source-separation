from argparse import ArgumentParser
from os import path

from thesis.data.toy import ToyData
from thesis.nn.models import WaveGlow
from thesis.train import train


def four_channel_unconditioned(data):
    batch_size, crop = 20, 2**11

    model = WaveGlow(channels=4, n_flows=10)
    loss_function = model.loss(σ=1.)

    train_loader = ToyData(f'{data}/toy_train.npy', crop) \
        .loader(batch_size)
    test_loader = ToyData(f'{data}/toy_test.npy', crop) \
        .loader(batch_size)
    return model, loss_function, train_loader, test_loader


def four_channel_conditioned(data):
    pass


def one_channel_unconditioned(data):
    pass


def one_channel_conditioned(data):
    pass


def main(args):
    if args.experiment == 'four_channel_unconditioned':
        func = four_channel_unconditioned
    elif args.experiment == 'four_channel_conditioned':
        func = four_channel_conditioned
    elif args.experiment == 'one_channel_unconditioned':
        func = one_channel_unconditioned
    elif args.experiment == 'one_channel_conditioned':
        func = one_channel_conditioned
    else:
        raise ValueError('Invalid experiment given.')

    model, loss_function, train_loader, test_loader = func(args.data)
    train(model=model, loss_function=loss_function, gpu=args.gpu,
          train_loader=train_loader, test_loader=test_loader,
          iterations=args.iterations, wandb=args.wandb)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('experiment', type=str, help='choose the experiment')
    parser.add_argument('--gpu', type=int, required=False, nargs='+',
                        help='The GPU ids to use. If unset, will use CPU.')
    parser.add_argument('--data', type=path.abspath, required=True,
                        help='The top-level directory of dataset.')
    parser.add_argument('-i', type=int, default=50000, dest='iterations',
                        help='Number of batches to train for.')
    parser.add_argument('-bs', type=int, default=20, dest='batch_size',
                        help='The batch size.')
    parser.add_argument('-wandb', action='store_true',
                        help='Logs to WandB.')
    main(parser.parse_args())
