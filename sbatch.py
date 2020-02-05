from datetime import datetime
import subprocess
from argparse import ArgumentParser
import os

from colorama import Fore

from train import EXPERIMENTS


def main(args):
    if args.experiment not in EXPERIMENTS:
        raise ValueError('Invalid experiment given.')

    p = 'gpu_short' if args.short else 'gpu'
    t = '0:05:00' if args.short else f'{args.hours}:00:00'

    c = {'job-name': args.experiment, 'ntasks': 1, 'cpus-per-task': 2,
         'ntasks-per-node': 1, 'time': t, 'mem': '10000M',
         'partition': p, 'gres': 'gpu:1'}

    s_c = '\n'.join(f'#SBATCH --{k}={v}' for k, v in c.items())

    f = f"#!/usr/bin/env bash\n\n{s_c}\n"
    f += "export LD_LIBRARY_PATH=/hpc/eb/Debian/cuDNN/7.4.2-CUDA-10.0.130/lib64:$LD_LIBRARY_PATH\n\n"""
    f += "cd /home/frankm/thesis\n"
    f += f"srun /home/frankm/.pyenv/shims/python3.7 train.py {args.experiment} --data=/home/frankm/data/toy/ -wandb --gpu 0"

    if args.short:
        f += '--batch_size=2'

    fn = '_temp.job'
    with open(fn, 'w') as fp:
        fp.write(f + '\n')
    os.makedirs('./log/', exist_ok=True)
    print(Fore.YELLOW + f'Written job file ./{fn}')
    o = f'{datetime.today():%y-%m-%d_%H}_{args.experiment}_{p}.out'
    rc = subprocess.call(["sbatch", fn, f'--output="./log/{o}"'])
    if rc == 0:
        os.remove(fn)
    exit(rc)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('--short', action='store_true')
    parser.add_argument('-t', type=int, default=5, dest='hours')
    main(parser.parse_args())
