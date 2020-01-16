from argparse import ArgumentParser
from os import makedirs
from os.path import basename, abspath
import os.path

import pandas as pd

from nsynth.sampling import load_model
from toy.ae import WavenetMultiAE
from toy.data import ToyDataSet
from toy.plot import plot_reconstruction, prepare_plot_freq_loss
from toy.vae import WavenetMultiVAE, ConditionalWavenetVQVAE


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
    args = parser.parse_args()

    makedirs('./figures', exist_ok=True)
    model = WavenetMultiVAE if args.vae else WavenetMultiAE
    fname = f'figures/{basename(args.weights)[:-3]}_freq_plot.npy'

    crop = 3 * 2 ** 10
    if not args.single:
        model = model(args.ns, 16, 64, 64, 10, 3, args.μ + 1, 1, False)
    else:
        model = ConditionalWavenetVQVAE(
            n_sources=args.ns, K=4, D=512, n_blocks=3, n_layers=10,
            encoder_width=64, decoder_width=32, in_channels=1,
            out_channels=args.μ + 1, device=args.device)
    model = load_model(args.weights, 'cpu', model)

    data = ToyDataSet(args.data, crop=crop)

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
