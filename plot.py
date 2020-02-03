from argparse import ArgumentParser
from os import makedirs
from os.path import abspath

from thesis.data.toy import ToyData
from thesis.io import load_model


def main(args):
    makedirs('./figures', exist_ok=True)
    model = load_model(args.weights, args.device)

    crop = 2 ** 11
    data = ToyData(args.data, crop)

    raise ValueError('Invalid Command given')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('command', type=str, help='show ∨ something')
    parser.add_argument('--weights', type=abspath, required=True)
    parser.add_argument('--data', type=abspath, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    main(parser.parse_args())
