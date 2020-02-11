#!/usr/bin/env python
from argparse import ArgumentParser
from os import makedirs
from os.path import abspath


def main(args):
    _ = args
    makedirs("./figures", exist_ok=True)
    raise ValueError("Invalid Command given")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", type=str, help="show ∨ something")
    parser.add_argument("--weights", type=abspath, required=True)
    parser.add_argument("--data", type=abspath, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    main(parser.parse_args())
