"""Inspect an existing v2/v3 classifier without retraining. Default: validation.

A readout override is an explicit post-training diagnostic, not proof that a
new objective has been trained. It never changes the saved checkpoint.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import torch
from visual_snn.data import make_loaders
from visual_snn.encoders import make_encoder
from visual_snn.engine import classifier_pass, seed_all
from visual_snn.model import make_model


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--readout', choices=['membrane', 'spike', 'paper'])
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--data-dir')
    p.add_argument('--download', action='store_true')
    p.add_argument('--fake-data', action='store_true', help='Software checks only; not detected in v2 checkpoints')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.threads < 1:
        p.error('threads must be positive')
    torch.set_num_threads(args.threads)
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if saved.get('implementation') not in ['last-vision-v2', 'last-vision-v3']:
        raise SystemExit('Requires a two-stage v2/v3 classifier checkpoint.')
    c = dict(saved['config'])
    training_readout = c.get('readout', 'paper')
    c['readout'] = args.readout or training_readout
    c['device'] = args.device
    if args.data_dir:
        c['data_dir'] = args.data_dir
    c['download'] = args.download
    seed_all(c['seed'])
    loaders, _, _ = make_loaders(c, fake=args.fake_data)
    encoder = make_encoder(c).to(args.device).eval()
    model = make_model(c).to(args.device).eval()
    encoder.load_state_dict(saved['encoder'])
    model.load_state_dict(saved['model'])
    result = classifier_pass(model, encoder, loaders[1 if args.split == 'validation' else 2], c,
                             encoding_seed=c['eval_seed'] + int(args.split == 'test'))
    result.update(training_readout=training_readout, evaluated_split=args.split,
                  checkpoint_epoch=saved['epoch'], fake_data=args.fake_data)
    print(json.dumps(result, indent=2))
