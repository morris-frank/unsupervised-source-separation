from argparse import ArgumentParser
from os import path, makedirs
from os.path import basename

import pandas as pd

from nsynth.sampling import load_model
from toy.ae import WavenetMultiAE
from toy.data import ToyDataSet
from toy.plot import plot_reconstruction, prepare_plot_freq_loss
from toy.vae import WavenetMultiVAE


def main():
    parser = ArgumentParser()
    parser.add_argument('--weights', type=path.abspath, required=True)
    parser.add_argument('--data', type=path.abspath, required=True)
    parser.add_argument('-ns', type=int, default=4)
    parser.add_argument('-μ', type=int, default=100)
    parser.add_argument('--mode', type=str, default='example')
    parser.add_argument('--length', type=int, default=500)
    parser.add_argument('--vae', action='store_true')
    args = parser.parse_args()

    makedirs('./figures', exist_ok=True)
    model = WavenetMultiVAE if args.vae else WavenetMultiAE
    fname = f'figures/{basename(args.weights)[:-3]}_freq_plot.npy'

    crop = 3 * 2 ** 10
    model = model(args.ns, 16, 64, 64, 10, 3, args.μ + 1, 1, False)
    model = load_model(args.weights, 'cpu', model)

    data = ToyDataSet(args.data, crop=crop)

    if args.mode.startswith('ex'):
        plot_reconstruction(model, data, args.ns, args.length)
    elif args.mode.startswith('freq'):
        dfs = [prepare_plot_freq_loss(model, data, args.ns, args.μ + 1, destroy)
               for destroy in [0, 0.5, 1]]
        df = pd.concat(dfs)
        pd.to_pickle(df, fname)


if __name__ == '__main__':
    main()
