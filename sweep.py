"""Launch matched rate, frozen-threshold LAST, and learned-threshold LAST runs."""
import argparse
import subprocess
import sys
from pathlib import Path


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['mnist', 'cifar100'], default='mnist')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--steps', nargs='+', type=int, default=[5])
    p.add_argument('--readouts', choices=['membrane', 'spike', 'paper'], nargs='+', default=['membrane'])
    p.add_argument('--root', default='runs/main')
    p.add_argument('--download', action='store_true')
    args = p.parse_args()
    for steps in args.steps:
        for seed in args.seeds:
            for readout in args.readouts:
                for method in ['rate', 'last_fixed', 'last']:
                    out = Path(args.root) / f'{args.dataset}_{method}_{readout}_t{steps}_s{seed}'
                    cmd = [sys.executable, str(Path(__file__).with_name('train.py')),
                           '--dataset', args.dataset, '--encoder', method, '--steps', str(steps),
                           '--seed', str(seed), '--readout', readout, '--out', str(out)]
                    if args.download:
                        cmd.append('--download')
                    subprocess.run(cmd, check=True)
