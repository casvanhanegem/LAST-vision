"""Train a LAST-PAIN visual extension or a matched rate baseline."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import torch
from visual_snn.engine import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version', action='version', version='LAST-vision-v3')
    p.add_argument('--dataset', choices=['mnist', 'cifar100'], default='mnist')
    p.add_argument('--encoder', choices=['last', 'last_fixed', 'rate'], default='last')
    p.add_argument('--readout', choices=['membrane', 'spike', 'paper'], default='membrane',
                   help='Aligned membrane/spike objective and prediction, or legacy paper-style comparison')
    p.add_argument('--encoder-checkpoint', help='Reuse compatible encoder_best.pt or best.pt; skip encoder pretraining')
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--encoder-hidden', type=int, default=3000)
    p.add_argument('--classifier-hidden', type=int, default=500)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--beta', type=float, default=0.99)
    p.add_argument('--slope', type=float, default=25.0)
    p.add_argument('--binary-cutoff', type=float, default=0.99)
    p.add_argument('--encoder-epochs', type=int, default=30)
    p.add_argument('--classifier-epochs', type=int, default=25)
    p.add_argument('--encoder-lr', type=float, default=0.005)
    p.add_argument('--classifier-lr', type=float, default=0.00075)
    p.add_argument('--count-weight', type=float, default=1.0)
    p.add_argument('--information-terms', choices=['all', 'repository'], default='all')
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--patience', type=int, default=0, help='Optional early stopping; 0 disables, best checkpoint still selected')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--eval-seed', type=int, default=12345)
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--download', action='store_true')
    p.add_argument('--fake-data', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--benchmark-repeats', type=int, default=20)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', required=True)
    c = vars(p.parse_args())
    for name in ['steps', 'encoder_hidden', 'classifier_hidden', 'encoder_epochs',
                 'classifier_epochs', 'batch_size', 'benchmark_repeats', 'threads']:
        if c[name] < 1:
            p.error(f'{name} must be positive')
    if not 0 < c['val_fraction'] < 1 or not 0 <= c['dropout'] < 1 or not 0 <= c['beta'] <= 1:
        p.error('Invalid validation fraction, dropout or beta')
    if not 0 < c['binary_cutoff'] < 1 or c['slope'] <= 0 or min(c['encoder_lr'], c['classifier_lr']) <= 0:
        p.error('Cutoff must be in (0,1); slope and learning rates must be positive')
    if min(c['count_weight'], c['weight_decay'], c['patience']) < 0 or (c['limit'] is not None and c['limit'] < 1):
        p.error('Invalid count weight, decay, patience or limit')
    if c['encoder'] == 'rate' and c['encoder_checkpoint']:
        p.error('Rate encoding has no encoder checkpoint to load')
    torch.set_num_threads(c.pop('threads'))
    run(c, c.pop('out'), c.pop('fake_data'))


if __name__ == '__main__':
    main()
