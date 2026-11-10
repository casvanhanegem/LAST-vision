# Method and provenance: extending LAST-PAIN to images

## V3 readout correction (26 October 2026)

Encoder methodology below is unchanged. V3 adds CE(mean output membrane,
labels) with membrane prediction and CE(summed output spikes, labels) with
spike-count prediction. Those variants select the highest validation accuracy,
with loss as tie-break. The older per-step membrane CE / spike prediction /
validation-loss selection is preserved as `--readout paper`. That label means
the prior source-style comparison, not exact full-paper parity.

Training/evaluation/prediction share readout definitions. Both accuracies and
silence are logged; there is no hidden fallback. Optional encoder-checkpoint
reuse isolates classifier changes while keeping encoder weights frozen. V3
does not repair or normalize the encoder proxy/count objective below; that
requires a separate validation-based investigation.

The intended contribution is a **visual extension / transfer study**: replace
the biosignal data with static visual data, retain the LAST encoder concept
and two-stage learning procedure, and measure the results. This is more than
a dataset loader switch, because image dimensions and temporal interpretation
must be made explicit.

## Preserved components and initial settings

| Component | V2 decision | Basis |
|---|---|---|
| Encoder | Two dense feature blocks with dropout, normalization and ReLU | Paper + public STAL encoder |
| Block order | Linear, dropout, batch norm, ReLU | Public implementation; paper's prose lists ReLU before normalization |
| Dense block widths | 3000, 3000 by default | Paper implementation details |
| Spike conversion | Repeat features; independent learned thresholds; soft sigmoid for pretraining | Paper + source |
| Threshold initialization | Uniform [0.4,0.6] | `utils/STAL.py` |
| Threshold bounds | Clamp to [0.001,1] after optimizer update | Source's bounds; clamping moved outside forward |
| Sigmoid slope | 25 | `utils/STAL.py` |
| Hard conversion | Probability >0.99 after encoder pretraining | Public `train.py` |
| Encoder optimizer | AdamW, learning rate .005 | Paper + source |
| Encoder training | 30 epochs, no classification labels in objective | Paper + source |
| Classifier | Dense hidden/output recurrent LIF layers | Paper + source |
| Hidden neurons | 500 | Paper + source |
| Output neurons | 10 / 100 | Necessary image-label adaptation |
| LIF beta / threshold | .99 / 1 | Paper + source |
| Reset / surrogate | Subtract threshold; snnTorch ATan derivative | Paper + public RLeaky default |
| Classifier optimizer | AdamW, learning rate .00075 | Paper + public helper |
| Classifier training | 25 epochs with frozen encoder | Paper's stated budget; source stages |
| Spike positions / simulation steps | 5 by default | Paper + source |
| Dropout | .5 | Paper + source |
| Weight decay | .01, configurable | Explicit AdamW library default, not separately reported by paper |

The source's current entry point has debugging/experiment overrides, including
50 classifier epochs and a batch-size override of 4. The paper states 25
classifier epochs and modality-specific batch sizes. This code uses the paper's
epoch budget and a shared image batch size of 32. Neither is claimed to be an
optimized MNIST/CIFAR-100 configuration.

## Necessary visual adaptations

1. Each grayscale/RGB image is flattened to N=C*H*W values. The image is one
   input window; biosignal sliding windows and sensor modalities are removed.
2. Hidden features are matched to N before repetition. Exact repeat/pooling is
   used where dimensions allow, adaptive average pooling otherwise. This avoids
   the source's hard divisibility constraints while retaining 3000-wide blocks.
   It changes the dimension-alignment operation; it is not paper-exact.
3. T spike positions are exposed as an explicit T,B,C,H,W tensor. They are
   artificial coding positions, not real recorded image-time dynamics.
4. The SRNN receives the same N-dimensional vector per coding position, with
   correct explicit temporal ordering. The source's flatten/reshape pipeline
   does not transparently preserve this ordering; its behavior is not copied.
5. There is one image classifier. No sEMG/IMU modality ensemble or random-forest
   meta classifier is used. These belong to the pain application, not the
   general encoder question.
6. Official image test sets and a fixed random training/validation split replace
   leave-one-subject-out validation. No subject identifiers exist in MNIST.
7. No class oversampling is used for these balanced image training datasets.
   Test labels do not affect pretraining, tuning or checkpoint selection.

Dense output features no longer have a guaranteed pixel-location meaning;
reshaping them to C,H,W is a tensor interface, not evidence of retained spatial
locality. The default SRNN flattens them again, so no convolution assumes
meaningful learned neighborhoods. A CNN feature extractor can be a later
separate visual ablation, not an unreported replacement in the baseline.

## Encoder objective: an important paper/source difference

Paper Eq.4 describes negative averaged mutual information at three levels and
an L1 reconstruction-like penalty. The released `EncoderLoss.py` instead uses:

- A differentiable **moment-based information proxy**
  J(X,Z)=mean(X*Z)*log2(mean(X*Z)/(mean(X)*mean(Z))).
- Weighted packing of soft spike positions with weights 1..T for that proxy.
- Per-sample absolute difference between the total soft-spike count and
  T times the summed input intensities, averaged across the batch.

That proxy is not a full joint/marginal mutual-information estimator. It can
be negative and depends on scaling. The count term encourages matching input
mass; it is not simply a penalty that always minimizes spikes. These semantics
matter especially for dense RGB images. `count_weight=1` preserves the released
loss's implicit unit coefficient; it is not claimed to resolve the paper's
unreported lambda value for Eq.4.

There is also an apparent source omission: in the two-layer encoder branch,
the code computes the first-layer proxy but does not add it before dividing
by three. V2 makes this choice explicit:

- `--information-terms all` (default): include input-to-Z1, input-to-Z2 and
  input-to-packed-spikes terms, divided by three. This corrects the omission
  to reflect the paper's three-level intent, while retaining the source proxy.
- `--information-terms repository`: reproduce the omission in that branch.

Both are source-derived objectives, **neither is a mathematically complete
implementation of the paper's MI objective**. A paper-faithful MI estimator and
literal Eq.4 would be a separate method variant requiring a justified estimator
and validation. Do not call this version an exact reproduction. Report the
selected objective mode and preferably ablate it before making scientific claims.

## Remaining protocol differences and limits

- Rate spikes are redrawn each training epoch and generated from a fixed
  evaluation stream, rather than saving one realization for every dataset
  before classifier training as the repository does. This is documented;
  add cached rate encoding if literal parity is required.
- Encoder inference recomputes deterministic features instead of caching
  binary events; encoder inference cost is included in runtime.
- Singleton training batches use BN running statistics, rather than skipping
  normalization entirely as the public encoder does.
- Validation chooses the lowest encoder objective and lowest classifier loss.
  Early-stopping patience is optional and explicit because it is not reported.
- Different T changes the summed classifier loss scale and threshold parameter
  count. Comparisons should state T; a T-matched experiment is the main baseline.
- LAST vs rate compares complete encoders with different parameter counts and
  feature-extraction capacity. It cannot isolate threshold learning alone.
  `last_fixed` pretrains the same feature architecture with frozen thresholds,
  providing a closer control. More controls may be needed for causal claims.
- The two-stage encoder does not receive classification labels. End-to-end
  classification fine-tuning would be a separate experiment, not the baseline.
- No full-dataset results, best image hyperparameters, neuromorphic energy
  measurements or demonstrated benefits are supplied with this project.

## Sources

Hens et al. (2025), *LAST-PAIN: Learning Adaptive Spike Thresholds for Low Back
Pain Biosignals Classification*, IEEE TNSRE 33:1038–1047.
https://doi.org/10.1109/TNSRE.2025.3546682

Published article text inspected via its full-text author/research copy:
https://www.researchgate.net/publication/389456333

Authors' public repository: https://github.com/freek1/emopain-stl
Relevant files: `utils/STAL.py`, `utils/EncoderLoss.py`,
`utils/RecurrentClassifier.py`, `utils/RateCoder.py`, `utils/helpers.py`, `train.py`.

This is an independently written implementation of the documented method
choices. Original authors' source files and pain data are not redistributed.
See `SOURCE_PROVENANCE.json` for retrieval identity and source-file hashes.
