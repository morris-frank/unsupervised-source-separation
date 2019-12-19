import pandas as pd
import seaborn as sns
from argparse import ArgumentParser
from itertools import permutations
from os import path

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
from tqdm import trange

from nsynth.sampling import load_model
from random import randint
from toy_train import ToyDataSet, WavenetMultiAE

mpl.use('TkAgg')


def toy2argmax(logits, ns):
    μ = 101
    signals = []
    for i in range(ns):
        j = i * μ
        signals.append(logits[:, j:j + μ, :].argmax(dim=1))
    return torch.cat(signals)


def reconstruction_compare_plot(mix, stems, pred, ns, length):
    fig, axs = plt.subplots(ns + 1, 2, sharex='all')
    for i in range(ns):
        axs[i, 0].plot(stems[i, 100:length], c='r')
        axs[i, 1].plot(pred[i, 100:length], c='b')
    axs[-1, 0].plot(mix[0, 0, 100:length], c='g')
    return fig


def plot_data_recon(model, data, ns, length):
    N = len(data)
    while True:
        mix, stems = data[randint(0, N)]
        mix = mix.unsqueeze(0)
        logits = model(mix)
        pred = toy2argmax(logits, ns)
        fig = reconstruction_compare_plot(mix, stems, pred, ns, length)
        plt.show()
        input()
        plt.close(fig)


def plot_freq_plot(model: WavenetMultiAE, data: ToyDataSet, ns: int, μ: int):
    d = {'shape': [], 'loss': [], 'periodicity': []}
    for n in trange(len(data.data)):
        mix, stems = data[n]
        prms = data.data[n]['params']

        logits = model(mix.unsqueeze(0))
        for i in range(ns):
            d['shape'].append(prms[i]['shape'])
            d['periodicity'].append(prms[i]['φ'])
            d['loss'].append(F.cross_entropy(logits[:, i * μ:(i + 1) * μ, :],
                                             stems[None, i, :]).item())
    df = pd.DataFrame(data=d)
    pd.to_pickle(df, 'freq_plot.npy')


def actually_plot_freq_plot(thres=4):
    df = pd.read_pickle('freq_plot.npy')
    sns.scatterplot(x='periodicity', y='loss', hue='shape', data=df)
    plt.show()


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
        plot_data_recon(model, data, args.ns, args.length)
    elif args.mode.startswith('freq'):
        plot_freq_plot(model, data, args.ns, args.μ + 1)


if __name__ == '__main__':
    main()
