from argparse import ArgumentParser
from os import path

from thesis.data.wrapper import map_dataset
from thesis.nn.models import WaveGlow, MultiRealNVP, ConditionalRealNVP
from thesis.train import train


def four_channel_unconditioned():
    batch_size = 12
    model = WaveGlow(channels=4, n_flows=15, wn_layers=12)  # rf: 2^11
    loss_function = model.loss(σ=1.)

    return model, loss_function, batch_size


def one_channel_unconditioned():
    batch_size = 14
    model = MultiRealNVP(channels=4, n_flows=15, wn_layers=10)  # rf: 2*2**9
    loss_function = model.loss(σ=1.)
    return model, loss_function, batch_size


def one_channel_conditioned():
    batch_size = 26
    model = ConditionalRealNVP(classes=4, n_flows=15, wn_layers=10,
                               wn_width=64)  # rf: 2*2**9
    loss_function = model.loss()
    return model, loss_function, batch_size


def main(args):
    if args.experiment not in EXPERIMENTS:
        raise ValueError('Invalid experiment given.')

    model, loss_function, batch_size = EXPERIMENTS[args.experiment]

    train_loader = map_dataset(model, args.data, 'train').loader(batch_size)
    test_loader = map_dataset(model, args.data, 'test').loader(batch_size)

    train(model=model, loss_function=loss_function, gpu=args.gpu,
          train_loader=train_loader, test_loader=test_loader,
          iterations=args.iterations, wandb=args.wandb,
          skip_test=args.skip_test)


EXPERIMENTS = {'4cu': four_channel_unconditioned,
               'four_channel_unconditioned': four_channel_unconditioned,
               '1cu': one_channel_unconditioned,
               'one_channel_unconditioned': one_channel_unconditioned,
               '1cc': one_channel_conditioned,
               'one_channel_conditioned': one_channel_conditioned}

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
