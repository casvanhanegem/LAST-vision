"""Unsupervised encoder pretraining followed by frozen-encoder SRNN training."""
import csv
import json
import platform
import hashlib
import random
import time
from pathlib import Path
import numpy as np
import torch
import snntorch
from torch.nn import functional as F
from .data import make_loaders
from .encoders import LastImageEncoder, make_encoder
from .losses import encoder_objective
from .model import make_model
from .readout import objective, scores, prefix_scores, improves


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def save_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def encoder_pass(encoder, loader, c, optimizer=None):
    encoder.train(optimizer is not None)
    total = 0
    accum = np.zeros(3)
    with torch.set_grad_enabled(optimizer is not None):
        for images, _ in loader:
            images = images.to(c['device'])
            p, x, z1, z2 = encoder.soft(images)
            loss, info, count = encoder_objective(p, x, z1, z2,
                c['count_weight'], c['information_terms'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite encoder objective')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                encoder.clamp_thresholds()
            accum += np.array([loss.item(), info.item(), count.item()]) * len(images)
            total += len(images)
    return dict(zip(['objective', 'information_proxy', 'count_match'], (accum / total).tolist()))


def classifier_loss(membranes, labels):
    # Public helper sums cross-entropy across simulation steps.
    return sum(F.cross_entropy(m, labels) for m in membranes)


def classifier_pass(model, encoder, loader, c, optimizer=None, encoding_seed=None):
    model.train(optimizer is not None)
    encoder.eval()
    generator = torch.Generator(device=c['device']).manual_seed(
        c['eval_seed'] if encoding_seed is None else encoding_seed)
    readout = c.get('readout', 'paper')
    total = correct = membrane_correct = spike_correct = no_output = tied_output = 0
    total_loss = input_count = hidden_count = output_count = events = slots = 0.0
    step_correct = torch.zeros(c['steps'], dtype=torch.int64)
    confusion = torch.zeros(model.fc2.out_features, model.fc2.out_features, dtype=torch.int64)
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(c['device']), labels.to(c['device'])
            with torch.no_grad():
                spikes = encoder(images, generator=generator)
            out_spikes, membranes, stats = model(spikes)
            loss = objective(out_spikes, membranes, labels, readout)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite classifier loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            counts = out_spikes.detach().sum(0)
            prediction = scores(out_spikes.detach(), membranes.detach(), readout).argmax(1)
            total += len(labels)
            correct += (prediction == labels).sum().item()
            membrane_correct += (membranes.detach().mean(0).argmax(1) == labels).sum().item()
            spike_correct += (counts.argmax(1) == labels).sum().item()
            no_output += (out_spikes.detach().sum((0, 2)) == 0).sum().item()
            tied_output += ((counts == counts.max(1, keepdim=True).values).sum(1) > 1).sum().item()
            total_loss += loss.item() * len(labels)
            input_count += spikes.sum().item()
            hidden_count += stats['hidden_spikes'].item()
            output_count += stats['output_spikes'].item()
            events += stats['synaptic_events_proxy'].item()
            slots += spikes.numel()
            step_correct += (prefix_scores(out_spikes.detach(), membranes.detach(), readout).argmax(2)
                             == labels[None]).sum(1).cpu()
            index = (labels * model.fc2.out_features + prediction).cpu()
            confusion += torch.bincount(index, minlength=confusion.numel()).reshape_as(confusion)
    return {'samples': total, 'accuracy': correct / total,
            'readout': readout, 'spike_readout_accuracy': spike_correct / total,
            'membrane_readout_accuracy': membrane_correct / total,
            'no_output_spike_fraction': no_output / total,
            'output_spike_tie_fraction': tied_output / total,
            'classifier_loss': total_loss / total,
            'input_spikes_per_sample': input_count / total, 'input_spike_density': input_count / slots,
            'hidden_spikes_per_sample': hidden_count / total, 'output_spikes_per_sample': output_count / total,
            'total_spikes_per_sample': (input_count + hidden_count + output_count) / total,
            'synaptic_events_proxy_per_sample': events / total,
            'accuracy_by_timestep': (step_correct.double() / total).tolist(),
            'confusion_matrix': confusion.tolist()}


def benchmark(model, encoder, loader, c):
    model.eval()
    encoder.eval()
    images = next(iter(loader))[0].to(c['device'])
    generator = torch.Generator(device=c['device']).manual_seed(100001)
    def sync():
        if torch.device(c['device']).type == 'cuda':
            torch.cuda.synchronize(c['device'])
    times = []
    with torch.inference_mode():
        for _ in range(5):
            model(encoder(images, generator=generator))
        for _ in range(c['benchmark_repeats']):
            sync()
            start = time.perf_counter()
            model(encoder(images, generator=generator))
            sync()
            times.append(1000 * (time.perf_counter() - start))
    return {'batch_size': len(images), 'median_batch_ms': float(np.median(times)),
            'median_amortized_ms_per_sample': float(np.median(times) / len(images)),
            'includes_dense_encoder': True, 'includes_spike_accounting': True,
            'excludes_loading_and_transfers': True}


def run(c, out, fake=False):
    c = dict(c)
    c.setdefault('readout', 'membrane')
    c.setdefault('encoder_checkpoint', None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    seed_all(c['seed'])
    (train, val, test), shape, split = make_loaders(c, fake)
    save_json(out / 'split.json', split)
    environment = {'python': platform.python_version(), 'torch': str(torch.__version__),
                   'snntorch': snntorch.__version__, 'device': c['device'],
                   'platform': platform.platform(), 'torch_threads': torch.get_num_threads()}
    save_json(out / 'environment.json', environment)
    encoder = make_encoder(c).to(c['device'])
    encoder_epoch = None
    if c['encoder_checkpoint']:
        if not isinstance(encoder, LastImageEncoder):
            raise ValueError('Encoder checkpoint only applies to LAST encoders')
        source = Path(c['encoder_checkpoint'])
        saved_encoder = torch.load(source, map_location=c['device'], weights_only=True)
        # v2 encoder-only checkpoints contain weights/epoch but no config.
        # For combined checkpoints, also validate non-weight settings.
        prior = saved_encoder.get('config')
        sidecar = source.with_name('config.json')
        if prior is None and sidecar.is_file():
            prior = json.loads(sidecar.read_text(encoding='utf-8'))
        if prior:
            for key in ['dataset', 'encoder', 'steps', 'encoder_hidden', 'dropout', 'slope',
                        'binary_cutoff', 'split_seed', 'val_fraction']:
                if prior.get(key) != c[key]:
                    raise ValueError(f'Encoder checkpoint setting mismatch for {key}: {prior.get(key)} vs {c[key]}')
            if prior.get('fake_data', False) and not fake:
                raise ValueError('Cannot reuse an explicitly fake-data encoder for a real-data experiment')
            save_json(out / 'encoder_source_config.json', prior)
            c['encoder_source_information_terms'] = prior.get('information_terms')
            c['encoder_source_count_weight'] = prior.get('count_weight')
        encoder.load_state_dict(saved_encoder['encoder'], strict=True)
        encoder_epoch = saved_encoder.get('encoder_epoch', saved_encoder.get('epoch'))
        hasher = hashlib.sha256()
        with source.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                hasher.update(block)
        digest = hasher.hexdigest()
        c['encoder_checkpoint_sha256'] = digest
        c['encoder_reused'] = True
        # Store a self-contained encoder checkpoint in this new run as well.
        torch.save({'encoder': encoder.state_dict(), 'epoch': encoder_epoch,
                    'config': c, 'implementation': 'last-vision-v3'}, out / 'encoder_best.pt')
        print(f'Reused encoder checkpoint (epoch {encoder_epoch}); skipping pretraining.', flush=True)
    elif isinstance(encoder, LastImageEncoder):
        c['encoder_reused'] = False
        optimizer = torch.optim.AdamW([p for p in encoder.parameters() if p.requires_grad],
                                      lr=c['encoder_lr'], weight_decay=c['weight_decay'])
        best, stale, rows = float('inf'), 0, []
        for epoch in range(1, c['encoder_epochs'] + 1):
            training = encoder_pass(encoder, train, c, optimizer)
            validation = encoder_pass(encoder, val, c)
            rows.append({'epoch': epoch, **{'train_' + k: v for k, v in training.items()},
                         **{'val_' + k: v for k, v in validation.items()}})
            save_csv(out / 'encoder_history.csv', rows)
            print(f"encoder {epoch}: val objective={validation['objective']:.4f}", flush=True)
            if validation['objective'] < best:
                best, stale = validation['objective'], 0
                torch.save({'encoder': encoder.state_dict(), 'epoch': epoch, 'config': c,
                            'implementation': 'last-vision-v3'}, out / 'encoder_best.pt')
            else:
                stale += 1
            if c['patience'] and stale >= c['patience']:
                break
        selected = torch.load(out / 'encoder_best.pt', map_location=c['device'], weights_only=True)
        encoder.load_state_dict(selected['encoder'])
        encoder_epoch = selected['epoch']
    if isinstance(encoder, LastImageEncoder):
        np.save(out / 'thresholds.npy', encoder.threshold.detach().cpu().numpy())
    save_json(out / 'config.json', {**c, 'fake_data': fake, 'implementation': 'last-vision-v3'})
    encoder.eval()
    encoder.requires_grad_(False)
    # Identical classifier initialization and shuffle start across encoders,
    # independent of how much encoder pretraining consumed the random stream.
    seed_all(c['seed'] + 9000)
    train.generator.manual_seed(c['seed'])
    model = make_model(c).to(c['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['classifier_lr'], weight_decay=c['weight_decay'])
    best, stale, rows = None, 0, []
    for epoch in range(1, c['classifier_epochs'] + 1):
        training = classifier_pass(model, encoder, train, c, optimizer, c['seed'] + epoch + 1000)
        validation = classifier_pass(model, encoder, val, c)
        rows.append({'epoch': epoch, 'train_loss': training['classifier_loss'],
                     'train_accuracy': training['accuracy'], 'val_loss': validation['classifier_loss'],
                     'val_accuracy': validation['accuracy'],
                     'train_spike_accuracy': training['spike_readout_accuracy'],
                     'train_membrane_accuracy': training['membrane_readout_accuracy'],
                     'train_no_output_fraction': training['no_output_spike_fraction'],
                     'val_spike_accuracy': validation['spike_readout_accuracy'],
                     'val_membrane_accuracy': validation['membrane_readout_accuracy'],
                     'val_no_output_fraction': validation['no_output_spike_fraction'],
                     'val_spike_tie_fraction': validation['output_spike_tie_fraction'],
                     'val_input_spike_density': validation['input_spike_density']})
        save_csv(out / 'classifier_history.csv', rows)
        print(f"classifier {epoch} [{c['readout']}]: val accuracy={validation['accuracy']:.4f} "
              f"spike={validation['spike_readout_accuracy']:.4f} "
              f"membrane={validation['membrane_readout_accuracy']:.4f} "
              f"silent={validation['no_output_spike_fraction']:.4f}", flush=True)
        if improves(validation, best, c['readout']):
            best, stale = dict(validation), 0
            torch.save({'encoder': encoder.state_dict(), 'model': model.state_dict(),
                        'config': c, 'epoch': epoch, 'encoder_epoch': encoder_epoch,
                        'best_val_accuracy': validation['accuracy'],
                        'selection': 'val_loss' if c['readout'] == 'paper' else 'val_accuracy_then_loss',
                        'implementation': 'last-vision-v3'}, out / 'best.pt')
        else:
            stale += 1
        if c['patience'] and stale >= c['patience']:
            break
    checkpoint = torch.load(out / 'best.pt', map_location=c['device'], weights_only=True)
    model.load_state_dict(checkpoint['model'])
    metrics = classifier_pass(model, encoder, test, c, encoding_seed=c['eval_seed'] + 1)
    n = shape[0] * shape[1] ** 2
    h, classes = c['classifier_hidden'], shape[2]
    metrics.update(best_epoch=checkpoint['epoch'], encoder_epoch=encoder_epoch,
                   best_val_accuracy=checkpoint['best_val_accuracy'], selection=checkpoint['selection'],
                   classifier_parameters=sum(p.numel() for p in model.parameters()),
                   encoder_parameters=sum(p.numel() for p in encoder.parameters()),
                   encoder_dense_weight_macs_per_sample=0 if c['encoder'] == 'rate' else
                       n * c['encoder_hidden'] + c['encoder_hidden'] ** 2,
                   classifier_dense_weight_macs_per_sample=c['steps'] * (n*h + h*h + h*classes + classes*classes))
    metrics['timing'] = benchmark(model, encoder, test, c)
    save_json(out / 'test_metrics.json', metrics)
    print(f"Saved {out}; test accuracy={metrics['accuracy']:.4f}", flush=True)
    return metrics
