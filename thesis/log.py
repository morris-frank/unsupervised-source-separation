from statistics import mean
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


class MonkeyWriter(object):
    """
    Monkey-patched empty version of SummaryWriter of Tensorboard
    """

    def add_scalar(self, tag, val, it):
        pass

    def add_figure(self, tag, val, it):
        pass

    def add_histogram(self, tag, val, it):
        pass

    def add_text(self, tag, val, it):
        pass


def log(writer: MonkeyWriter, it: int, values: Dict):
    """
    Logs the given key value pairs. Writes to CLI and to a given writer.
    :param writer: The writer (SummaryWriter of TensorBoard)
    :param it: current global iteration
    :param values: Dict of tag⇒values to log
    """
    mess = f'it={it:>10}\t'

    for tag, val in values.items():
        if isinstance(val, plt.Figure):
            writer.add_figure(tag, val, it)

        elif isinstance(val, list) and len(val) > 0:
            try:
                writer.add_histogram(tag, np.array(val), it)
            except ValueError:
                print('\tEmpty histogram')
            mean_tag, mean_val = f'{tag}', mean(val)
            mess += f'\t{mean_tag}:{mean_val:.3e}'
            writer.add_scalar(mean_tag, mean_val, it)

        elif isinstance(val, str):
            writer.add_text(tag, val, it)

        else:
            mess += f'\t{tag}:{val:.3e}'
            writer.add_scalar(tag, val, it)
    print(mess)
