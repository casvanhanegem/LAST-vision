"""Explicit classifier objectives/readouts, shared by train/evaluate/predict."""
import torch
from torch.nn import functional as F


def scores(spikes, membranes, readout):
    if readout == 'membrane':
        return membranes.mean(0)
    if readout in ('spike', 'paper'):
        return spikes.sum(0)
    raise ValueError(f'Unknown readout: {readout}')


def objective(spikes, membranes, labels, readout):
    if readout == 'paper':
        # Preserved v2/public-helper protocol, including its readout mismatch.
        return sum(F.cross_entropy(m, labels) for m in membranes)
    return F.cross_entropy(scores(spikes, membranes, readout), labels)


def prefix_scores(spikes, membranes, readout):
    if readout == 'membrane':
        divisor = torch.arange(1, len(membranes) + 1, device=membranes.device,
                               dtype=membranes.dtype)[:, None, None]
        return membranes.cumsum(0) / divisor
    if readout in ('spike', 'paper'):
        return spikes.cumsum(0)
    raise ValueError(readout)


def improves(candidate, incumbent, readout):
    """Aligned variants: validation accuracy, loss tie-break. Paper: loss."""
    if incumbent is None:
        return True
    if readout == 'paper':
        return candidate['classifier_loss'] < incumbent['classifier_loss']
    return (candidate['accuracy'], -candidate['classifier_loss']) > (
        incumbent['accuracy'], -incumbent['classifier_loss'])
