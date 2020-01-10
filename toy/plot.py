from random import randint

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.nn import functional as F
from tqdm import trange

from .ae import WavenetMultiAE
from .data import ToyDataSet
from .functional import toy2argmax

mpl.use('TkAgg')


def fig_reconstruction(mix, stems, pred, ns, length):
    fig, axs = plt.subplots(ns + 1, 2, sharex='all')
    for i in range(ns):
        axs[i, 0].plot(stems[i, 100:length], c='r')
        axs[i, 1].plot(pred[i, 100:length], c='b')
    axs[-1, 0].plot(mix[0, 0, 100:length], c='g')
    return fig


def meta_forward(model, mix, ns, single, destroy=0.):
    with torch.no_grad():
        if single:
            logits = [
                model.test_forward(mix, torch.tensor([i]), destroy=destroy)
                for i in range(ns)]
            logits = torch.cat(logits, 1)
        else:
            logits = model.test_forward(mix, destroy=destroy)
    return logits


def plot_reconstruction(model, data, ns, length, single=False):
    N = len(data)
    model.eval()
    while True:
        mix, stems = data[randint(0, N)]
        mix = mix.unsqueeze(0)
        logits = meta_forward(model, mix, ns, single)
        pred = toy2argmax(logits, ns)
        fig = fig_reconstruction(mix, stems, pred, ns, length)
        plt.show()
        input()
        plt.close(fig)


def prepare_plot_freq_loss(model: WavenetMultiAE, data: ToyDataSet, ns: int,
                           μ: int, destroy: float = 0.,
                           single: bool = False) -> pd.DataFrame:
    d = {'shape': [], 'loss': [], 'periodicity': [], 'destroy': []}
    model.eval()
    for n in trange(len(data.data)):
        mix, stems = data[n]
        prms = data.data[n]['params']

        logits = meta_forward(model, mix, ns, single, destroy)
        for i in range(ns):
            d['shape'].append(prms[i]['shape'])
            d['destroy'].append(destroy)
            d['periodicity'].append(prms[i]['φ'])
            d['loss'].append(F.cross_entropy(logits[:, i * μ:(i + 1) * μ, :],
                                             stems[None, i, :]).item())
    df = pd.DataFrame(data=d)
    return df


def plot_freq_loss(thres=4):
    df = pd.read_pickle('freq_plot.npy')
    sns.scatterplot(x='periodicity', y='loss', hue='shape', data=df)
    plt.show()
