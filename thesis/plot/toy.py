import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from ..data.toy import ToyData
from ..functional import multi_argmax

mpl.use('TkAgg')


def fig_summary(fname: str):
    df = pd.read_pickle(fname)
    df.destroy = df.destroy.astype('category')
    zer = df[df.destroy == 0.0].iloc[0].destroy

    fig = plt.figure(figsize=(5, 8))

    ax1 = fig.add_subplot(3, 2, 1)
    plt.title('Influence of the latent embedding')
    sns.scatterplot(x='periodicity', y='loss', hue='destroy', data=df,
                    palette=['r', 'g', 'b'], ax=ax1)

    ax2 = fig.add_subplot(3, 2, 3)
    plt.title('Mean loss over different destroy and source shapes')
    sns.boxplot(x="destroy", y="loss", hue="shape", data=df, ax=ax2)

    ax3 = fig.add_subplot(3, 2, 5)
    plt.title('Loss over shapes @ no destroy ')
    sns.scatterplot(x='periodicity', y='loss', hue='shape',
                    data=df[df.destroy == zer], ax=ax3)

    ax4 = fig.add_subplot(2, 2, 2)
    plt.title('Mean period/loss for a sample @ no destroy')
    rdf = df[df.destroy == zer].groupby('n').agg('mean')
    sns.relplot(x="periodicity", y="loss", data=rdf, ax=ax4)

    ax5 = fig.add_subplot(2, 2, 4)
    plt.title('Sum period/loss for a sample @ no destroy')
    rdf = df[df.destroy == zer].groupby('n').agg('sum')
    sns.relplot(x="periodicity", y="loss", data=rdf, ax=ax5)

    return fig


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
                model.test_forward(mix, torch.tensor([i]),
                                   destroy=destroy)
                for i in range(ns)]
            logits = torch.cat(logits, 1)
        else:
            logits = model.test_forward(mix, destroy=destroy)
    return logits


def plot_reconstruction(model, data, ns, length, single=False):
    model.eval()
    for i, (x, stems) in enumerate(data):
        if isinstance(x, tuple):
            mix, label = x
            mix = mix.unsqueeze(0)
            x = (mix, label)
        else:
            mix = x
            mix = mix.unsqueeze(0)
            x = mix
        logits = meta_forward(model, x, ns, single)
        pred = multi_argmax(logits, ns)
        fig = fig_reconstruction(mix, stems, pred, ns, length)
        fig.savefig(f'./figures/{type(model).__name__}_{i}.png')
        plt.show()
        input()
        plt.close(fig)


def prepare_plot_freq_loss(model: nn.Module, data: ToyData, ns: int,
                           μ: int, destroy: float = 0.,
                           single: bool = False,
                           device: str = 'cpu') -> pd.DataFrame:
    d = {'n': [], 'shape': [], 'loss': [], 'periodicity': [], 'destroy': []}
    model = model.to(device)
    model.eval()
    for n in trange(len(data.data)):
        mix, stems = data[n]
        mix = mix.unsqueeze(0)
        prms = data.data[n]['params']

        logits = meta_forward(model, mix.to(device), ns, single, destroy)
        logits = logits.cpu()
        for i in range(ns):
            d['n'].append(n)
            d['shape'].append(prms[i]['shape'])
            d['destroy'].append(destroy)
            d['periodicity'].append(prms[i]['φ'])
            d['loss'].append(F.cross_entropy(logits[:, i * μ:(i + 1) * μ, :],
                                             stems[None, i, :]).item())
    df = pd.DataFrame(data=d)
    return df


def plot_freq_loss(fname: str):
    df = pd.read_pickle(fname)
    df.destroy = df.destroy.astype('category')
    # zer = df.destroy.iloc[0]
    sns.scatterplot(x='periodicity', y='loss', hue='shape', data=df)
    plt.show()
