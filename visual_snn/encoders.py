"""Independent implementation informed by LAST-PAIN and its authors' STAL code.

Images replace biosignal windows. Binary events have explicit [T,B,C,H,W]
ordering; we do not reuse the repository's ambiguous flattened spike ordering.
"""
import torch
from torch import nn
from torch.nn import functional as F


def align_features(z, count):
    """Original repeat/pool where possible; adaptive pooling for arbitrary sizes."""
    if z.shape[1] == count:
        return z
    if z.shape[1] < count and count % z.shape[1] == 0:
        return z.repeat_interleave(count // z.shape[1], dim=1)
    return F.adaptive_avg_pool1d(z.unsqueeze(1), count).squeeze(1)


class LastImageEncoder(nn.Module):
    """Two dense feature blocks followed by repeated features and thresholds.

    Features are matched to C*H*W. Matching before spike repetition is an
    explicit image adaptation that removes the repository's divisibility
    restriction. Dense output positions are learned features, not raw pixels.
    """
    def __init__(self, channels, size, steps=5, hidden=3000, dropout=0.5,
                 slope=25.0, binary_cutoff=0.99, learn_thresholds=True):
        super().__init__()
        self.channels, self.size, self.steps = channels, size, steps
        self.count = channels * size * size
        self.slope, self.binary_cutoff = slope, binary_cutoff
        self.linear1 = nn.Linear(self.count, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.bn1, self.bn2 = nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden)
        # Source code: uniform(0.4,0.6), one threshold per repeated feature.
        self.threshold = nn.Parameter(torch.empty(steps, self.count).uniform_(0.4, 0.6),
                                      requires_grad=learn_thresholds)

    def _block(self, x, linear, dropout, bn):
        x = dropout(linear(x))
        # BN needs >1 sample when training; use its running statistics otherwise.
        if self.training and len(x) == 1:
            x = F.batch_norm(x, bn.running_mean, bn.running_var,
                             bn.weight, bn.bias, training=False, eps=bn.eps)
        else:
            x = bn(x)
        return F.relu(x)

    def soft(self, images):
        flat = images.flatten(1)
        z1 = self._block(flat, self.linear1, self.drop1, self.bn1)
        z2 = self._block(z1, self.linear2, self.drop2, self.bn2)
        aligned1, aligned2 = align_features(z1, self.count), align_features(z2, self.count)
        probabilities = torch.sigmoid(self.slope * (aligned2[None] - self.threshold[:, None]))
        return probabilities, flat, aligned1, aligned2

    @torch.no_grad()
    def clamp_thresholds(self):
        self.threshold.clamp_(0.001, 1.0)

    def forward(self, images, generator=None):
        probabilities, *_ = self.soft(images)
        spikes = (probabilities > self.binary_cutoff).to(images.dtype)
        return spikes.reshape(self.steps, len(images), self.channels, self.size, self.size)


class RateEncoder(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.steps = steps

    def forward(self, images, generator=None):
        random = torch.rand((self.steps, *images.shape), device=images.device,
                            dtype=images.dtype, generator=generator)
        return (random < images[None]).to(images.dtype)


def make_encoder(c):
    channels, size = (1, 28) if c['dataset'] == 'mnist' else (3, 32)
    if c['encoder'] == 'rate':
        return RateEncoder(c['steps'])
    return LastImageEncoder(channels, size, c['steps'], c['encoder_hidden'],
                            c['dropout'], c['slope'], c['binary_cutoff'],
                            learn_thresholds=c['encoder'] != 'last_fixed')
