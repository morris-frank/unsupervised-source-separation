from argparse import ArgumentParser
from os.path import abspath

import matplotlib.pyplot as plt

from thesis.plot.toy import fig_summary


def main():
    parser = ArgumentParser()
    parser.add_argument('-p', type=abspath, required=True)
    args = parser.parse_args()

    _ = fig_summary(args.p)

    plt.show()


if __name__ == '__main__':
    main()
