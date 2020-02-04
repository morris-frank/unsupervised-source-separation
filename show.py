import torch
from argparse import ArgumentParser
from os.path import abspath

from matplotlib import pyplot as plt

from thesis.data.wrapper import map_dataset
from thesis.io import load_model


def main(args):
    if args.command == 'example':
        from thesis.plot.toy import example_reconstruction
        model = load_model(args.weights, args.device)
        model.eval()
        data = map_dataset(model, args.data, 'test')

        with torch.no_grad():
            for fig in example_reconstruction(model, data):
                fig.show()
                input()
                plt.close(fig)

    elif args.command == 'z-sample':
        from thesis.plot.toy import z_example_reconstruction
        model = load_model(args.weights, args.device)
        model.eval()
        data = map_dataset(model, args.data, 'train')
        with torch.no_grad():
            for fig in z_example_reconstruction(model, data):
                fig.show()
                input()
                plt.close(fig)

    else:
        raise ValueError('Invalid command given')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('command', type=str)
    parser.add_argument('--weights', type=abspath)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--data', type=abspath)
    main(parser.parse_args())
