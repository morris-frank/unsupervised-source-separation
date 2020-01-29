from argparse import ArgumentParser
from os import path

from thesis.data.toy import ToyData
from thesis.nn.models import WaveGlow
from thesis.train import train


def main(args):
    args.epochs = 50000
    crop = 2 ** 11

    model = WaveGlow(channels=4, n_flows=10)
    loss_function = model.loss(σ=1.)

    train_loader = ToyData(f'{args.data}/toy_train.npy', crop) \
        .loader(args.batch_size)
    test_loader = ToyData(f'{args.data}/toy_test.npy', crop) \
        .loader(args.batch_size)

    train(model=model, loss_function=loss_function, gpu=args.gpu,
          train_loader=train_loader, test_loader=test_loader,
          iterations=args.iterations, wandb=args.wandb)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, required=False, nargs='+',
                        help='The GPU ids to use. If unset, will use CPU.')
    parser.add_argument('--data', type=path.abspath, required=True,
                        help='The top-level directory of dataset.')
    parser.add_argument('-i', type=int, default=50000, dest='iterations',
                        help='Number of batches to train for.')
    parser.add_argument('-bs', type=int, default=32, dest='batch_size',
                        help='The batch size.')
    parser.add_argument('-wandb', action='store_true',
                        help='Logs to WandB.')
    main(parser.parse_args())
