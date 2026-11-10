# V3 validation

26 October 2026; Python 3.12, torch 2.5.1+cpu, torchvision 0.20.1+cpu,
snntorch 0.9.4, CPU. Core versions match `requirements-tested.txt`.

Fourteen tests pass: the eight v2 encoder/data/recurrent tests and six new
readout tests. New tests check objective/score and prefix agreement, spike-loss
gradients despite silent outputs, preserved legacy behavior, metric-aligned
checkpoint selection, and membrane prediction despite silent spike counts.

Passing tests establishes behavior, not high accuracy. Full MNIST v3 training,
encoder loss rebalancing, CUDA and scientific superiority remain unverified.
The code does not claim to reproduce the user's private trained weights or
recover their accuracy without those weights. Local encoder reuse is supported.

Rechecked all fourteen tests and completed a one-epoch-per-stage synthetic-data
membrane-readout smoke run. This checks execution only, not learning quality.
