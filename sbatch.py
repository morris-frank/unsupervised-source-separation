import subprocess
from argparse import ArgumentParser

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
    f += f"srun /home/frankm/.pyenv/shims/python3.7 train.py {args.experiment} --data=/home/frankm/data/toy/ -wandb --gpu 0\n"

    fn = '_temp.job'
    with open(fn, 'w') as fp:
        fp.write(f)
    print(Fore.YELLOW + f'Written job file ./{fn}')
    subprocess.call(["sbatch", fn])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('--short', action='store_true')
    parser.add_argument('-t', type=int, default=5, dest='hours')
    main(parser.parse_args())
