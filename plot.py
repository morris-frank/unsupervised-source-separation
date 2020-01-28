import os.path
from argparse import ArgumentParser
from os import makedirs
from os.path import basename, abspath

import pandas as pd

from thesis.data.toy import ToyDataSequential
from thesis.io import load_model
from thesis.plot.toy import prepare_plot_freq_loss, plot_reconstruction

"""
model = ConditionalWavenetVQVAE(
            n_sources=args.ns, K=4, D=512, n_blocks=3, n_layers=10,
            encoder_width=64, decoder_width=32, in_channels=1,
            out_channels=args.μ + 1)
"""


def main():
    parser = ArgumentParser()
    parser.add_argument('--weights', type=abspath, required=True)
    parser.add_argument('--data', type=abspath, required=True)
    parser.add_argument('-ns', type=int, default=4)
    parser.add_argument('-μ', type=int, default=100)
    parser.add_argument('--mode', type=str, default='example')
    parser.add_argument('--length', type=int, default=500)
    parser.add_argument('--vae', action='store_true')
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--destroy', type=float, default=0.5)
    parser.add_argument('--steps', type=int, default=5)
    args = parser.parse_args()

    makedirs('./figures', exist_ok=True)
    fname = f'figures/{basename(args.weights)[:-3]}_freq_plot.npy'
    model = load_model(args.weights, 'cpu')

    crop = 3 * 2 ** 10
    data = ToyDataSequential(args.data, μ=args.μ, crop=crop, steps=args.steps,
                             batch_size=1)

    if args.mode.startswith('ex'):
        plot_reconstruction(model, data, args.ns, args.length,
                            single=args.single)
    elif args.mode.startswith('freq'):
        df = prepare_plot_freq_loss(model, data, args.ns, args.μ + 1,
                                    args.destroy,
                                    single=args.single, device=args.device)
        if os.path.exists(fname):
            _df = pd.read_pickle(fname)
            df = pd.concat([df, _df])
        pd.to_pickle(df, fname)


if __name__ == '__main__':
    main()
