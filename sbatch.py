#!/usr/bin/env python
import os
import subprocess
from argparse import ArgumentParser
from datetime import datetime

from colorama import Fore

from train import EXPERIMENTS


def main(args):
    if args.file not in ("make", "train"):
        raise ValueError("Invalid file given")
    if args.file == "train" and args.experiment not in EXPERIMENTS:
        raise ValueError("Invalid experiment given.")

    p = "gpu_short" if args.short else "gpu_shared"
    t = "0:30:00" if args.short else f"{args.hours}:00:00"

    name = args.experiment
    if args.k:
        name += "_" + args.k

    c = {
        "job-name": name,
        "cpus-per-task": 2,
        "time": t,
        "mem": "10000M",
        "partition": p,
        "gres": "gpu:1",
        "output": f"./log/{datetime.today():%b%d-%H%M}_%x_{p}.out",
    }

    f = f"#!/usr/bin/env bash\n\n"
    f += "\n".join(f"#SBATCH --{k}={v}" for k, v in c.items()) + "\n"
    f += (
        'export PATH="/home/frankm/.local/bin:$PATH"\n'
        "export LD_LIBRARY_PATH="
        "/hpc/eb/Debian/cuDNN/7.4.2-CUDA-10.0.130/lib64:$LD_LIBRARY_PATH\n\n"
        "export LC_ALL=en_US.utf8\n"
        'export LANG="$LC_ALL"\n'
    )
    f += "cd /home/frankm/thesis\n"

    f += (
        f"srun /home/frankm/.pyenv/shims/python3.7 {args.file}.py "
        f"{args.experiment} --data=/home/frankm/data/toy/"
    )

    if args.file == "train":
        f += f" --batch_size={args.batch_size} -wandb --gpu 0"

    if args.debug:
        f += " -debug"

    if args.musdb:
        f += " -musdb"

    if args.k:
        f += f" -k {args.k}"

    fn = "_temp.job"
    with open(fn, "w") as fp:
        fp.write(f + "\n")
    os.makedirs("./log/", exist_ok=True)
    print(Fore.YELLOW + f"Written job file ./{fn}")
    rc = subprocess.call(["sbatch", fn])
    if rc == 0:
        os.remove(fn)
    exit(rc)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("--short", action="store_true")
    parser.add_argument("-t", type=int, default=5, dest="hours")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("-f", type=str, default="train", dest="file")
    parser.add_argument("-k", type=str, required=False)
    parser.add_argument("-debug", action="store_true")
    parser.add_argument("-musdb", action="store_true")
    main(parser.parse_args())
