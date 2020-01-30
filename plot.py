from argparse import ArgumentParser
from argparse import ArgumentParser
from os import makedirs
from os.path import abspath

from thesis.data.toy import ToyData
from thesis.io import load_model
from thesis.plot.toy import plot_reconstruction


def main(args):
    makedirs('./figures', exist_ok=True)
    # fname = f'figures/{basename(args.weights)[:-3]}_freq_plot.npy'
    model = load_model(args.weights, args.device)

    crop = 2 ** 11
    data = ToyData(args.data, crop)

    if args.command == 'show':
        plot_reconstruction(model, data)
    else:
        raise ValueError('Invalid Command given')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('command', type=str, help='show ∨ something')
    parser.add_argument('--weights', type=abspath, required=True)
    parser.add_argument('--data', type=abspath, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    main(parser.parse_args())
