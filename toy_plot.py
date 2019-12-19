from argparse import ArgumentParser
from os import path

from nsynth.sampling import load_model
from toy.plot import plot_reconstruction, prepare_plot_freq_loss
from toy_train import ToyDataSet, WavenetMultiAE


def main():
    parser = ArgumentParser()
    parser.add_argument('--weights', type=path.abspath, required=True)
    parser.add_argument('--data', type=path.abspath, required=True)
    parser.add_argument('-ns', type=int, default=4)
    parser.add_argument('-μ', type=int, default=100)
    parser.add_argument('--mode', type=str, default='example')
    parser.add_argument('--length', type=int, default=500)
    args = parser.parse_args()

    crop = 3 * 2 ** 10
    model = WavenetMultiAE(args.ns, 16, 64, 64, 10, 3, args.μ + 1, 1, False)
    model = load_model(args.weights, 'cpu', model)

    data = ToyDataSet(args.data, crop=crop)

    if args.mode.startswith('ex'):
        plot_reconstruction(model, data, args.ns, args.length)
    elif args.mode.startswith('freq'):
        prepare_plot_freq_loss(model, data, args.ns, args.μ + 1)


if __name__ == '__main__':
    main()
