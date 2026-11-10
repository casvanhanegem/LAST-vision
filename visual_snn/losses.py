"""Documented interpretation of the public implementation's encoder objective.

This retains the author's moment-based information proxy and population-count
matching term. It is NOT a full mutual-information estimator or Eq.4 verbatim.
"""
import torch


def information_proxy(x, z):
    joint = (x * z).mean().clamp_min(1e-12)
    product = (x.mean() * z.mean()).clamp_min(1e-12)
    return joint * torch.log2(joint / product)


def encoder_objective(probabilities, x, z1, z2, count_weight=1.0,
                      information_terms='all'):
    # probabilities: T,B,N. Weights 1..T, as in the repository loss.
    weights = torch.arange(1, len(probabilities) + 1,
                           device=x.device, dtype=x.dtype)[:, None, None]
    packed = (probabilities * weights).sum(0)
    info = information_proxy(x, packed) + information_proxy(x, z2)
    if information_terms == 'all':
        info = info + information_proxy(x, z1)
    elif information_terms != 'repository':
        raise ValueError(information_terms)
    # The repository divides by 3 but accidentally omits Z1 in its two-layer
    # branch. 'repository' preserves that; 'all' explicitly includes all terms.
    info = info / 3
    count_match = (probabilities.sum((0, 2)) - len(probabilities) * x.sum(1)).abs().mean()
    return -info + count_weight * count_match, info, count_match
