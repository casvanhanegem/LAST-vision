# LAST visual extension — version 3

V3 addresses the classifier readout mismatch observed in a full MNIST run.
It does not claim to fix every limitation or to establish high image accuracy.
The encoder, its two-stage protocol and its source-derived loss are unchanged.

## Choose an explicit classifier protocol

| `--readout` | Training loss | Prediction | Checkpoint selection |
|---|---|---|---|
| `membrane` (new default) | Cross-entropy of mean output membrane potentials | Highest mean membrane potential | Highest validation membrane accuracy, loss tie-break |
| `spike` | Cross-entropy of summed output spikes using surrogate gradients | Highest spike count | Highest validation spike accuracy, loss tie-break |
| `paper` | Sum of per-step membrane cross-entropies | Highest spike count | Lowest validation loss |

`paper` preserves the previous paper/source-style protocol as a comparison; it
does not mean exact replication of the whole paper. The aligned variants are
explicit visual-study changes. The same recurrent LIF architecture remains.
No fallback silently switches spike predictions to membrane scores. In membrane
mode output spikes may still be silent, but class decisions use analogue state.
That is a readout change, not proof that output spike activity has improved.

## Install and check the version

Extract into a new folder. Open a terminal inside `last_vision_v3`, containing
`train.py`. Keep the old results/checkpoints. On Windows PowerShell:

```powershell
python -m pip install -r requirements.txt
python train.py --version
python -m unittest discover -s tests -v
```

The version command must print `LAST-vision-v3`. Use a compatible torch/
torchvision pair for your CPU/CUDA installation. Tested core versions are
recorded separately; they are not a full dependency lock.

## First experiment: reuse your existing trained encoder

If the old project is in `C:\Users\casva\Desktop\SNN-two`, use:

```powershell
python train.py --dataset mnist --encoder last --readout membrane --encoder-checkpoint "C:\Users\casva\Desktop\SNN-two\runs\mnist_last_full\encoder_best.pt" --data-dir "C:\Users\casva\Desktop\SNN-two\data" --out runs/mnist_membrane
```

This loads and freezes the selected encoder, skips its 30 training epochs and
trains a fresh classifier for the default 25 epochs. It is not classifier
training resume. The default encoder width 3000, classifier width 500, five
steps and other encoder settings match the user's default previous run.

To train an aligned spike-count variant using the same encoder:

```powershell
python train.py --dataset mnist --encoder last --readout spike --encoder-checkpoint "C:\Users\casva\Desktop\SNN-two\runs\mnist_last_full\encoder_best.pt" --data-dir "C:\Users\casva\Desktop\SNN-two\data" --out runs/mnist_spike
```

Use `--readout paper` with a new output folder for a matched legacy comparison.
Classifier initialization and initial training shuffle reset from the same seed
for every variant. Choose the research protocol using validation data, not a
sequence of test-set comparisons.

For a v2 encoder-only checkpoint, a sibling `config.json` is used if available.
Combined/v3 checkpoints embed configuration. Known encoder settings and data
split settings must match; the source config and file hash are recorded. Without
metadata, shapes are checked but you must supply matching dataset, width, steps,
dropout, slope and cutoff. Explicitly fake-data encoders cannot be reused in a
real-data run. Reused source settings are preserved in `encoder_source_config.json`.
Do not load a learned-threshold encoder as a valid frozen-threshold pretraining
control; reuse only the matching encoder condition.

## Progress and diagnostics

```text
classifier 1 [membrane]: val accuracy=... spike=... membrane=... silent=...
```

`val accuracy` is primary accuracy for the selected protocol. `spike` and
`membrane` show the two decisions separately. `silent` is the fraction of
samples with no output spikes in any class or timestep. Values are fractions.
The history CSV records both train/validation accuracies and silence, plus
validation spike ties and input spike density.

To inspect an existing classifier without training (default: validation split):

```powershell
python evaluate.py --checkpoint "C:\Users\casva\Desktop\SNN-two\runs\mnist_last_full\best.pt" --readout membrane --split validation --data-dir "C:\Users\casva\Desktop\SNN-two\data"
```

A readout override is an explicit post-training diagnostic; it does not mean
the old checkpoint was trained with a new objective. V2's training protocol
defaults to `paper`. This script never changes saved weights.

## Smoke test and training from scratch

```powershell
python train.py --fake-data --encoder-hidden 16 --classifier-hidden 16 --encoder-epochs 1 --classifier-epochs 1 --benchmark-repeats 2 --out runs/smoke
python train.py --dataset mnist --encoder last --readout membrane --download --out runs/mnist_new_encoder
python train.py --dataset mnist --encoder rate --readout membrane --out runs/mnist_rate
```

Random-data accuracy has no scientific meaning. `--limit 256` with short epoch
budgets is another software check; summaries exclude limited/fake data.
LAST still uses two dense blocks, sigmoid thresholds, unsupervised encoder
training then binary conversion and frozen-encoder classifier training. Defaults
remain encoder widths 3000/3000, recurrent hidden width 500, five coding steps,
learning rates .005/.00075 and stage budgets 30/25 epochs.

The encoder objective is deliberately unchanged. The observed scalar magnitude
gap between count matching and the information proxy does not establish their
gradient balance. No arbitrary loss-weight fix has been silently applied.
`METHOD.md` records the source-derived objective and its limitations. A full
mutual-information estimator or normalized count loss would be separate variants.

## Repeated experiments and saved files

```powershell
python sweep.py --dataset mnist --readouts membrane --seeds 0 1 2 --download --root runs/study
python summarize.py --root runs/study --out analysis/study
```

All encoder conditions use the same classifier/readout protocol. Rate skips
encoder pretraining. LAST / `last_fixed` pretrain features with learned / frozen
thresholds. Multiple `--readouts` create separate experiments; the sweep
repeats deterministic pretraining rather than automatically caching encoders.

Each run saves configuration, environment, split indices, histories, encoder
checkpoint/thresholds for LAST, combined `best.pt`, and `test_metrics.json`.
New output folders are required; previous results are not overwritten. Encoder
reuse saves a self-contained encoder copy in the new run. Exact optimizer/RNG
resume is not implemented. Summary groups distinguish readout protocols and
fresh versus reused encoder runs; plot labels include readout. Reused checkpoint
paths/hashes are retained in per-run outputs, not used as seed grouping keys.

`predict.py --checkpoint ... --image digit.png` uses the checkpoint's protocol;
v2 defaults to spike decisions, and an override must be explicit. MNIST expects
centered white digits on black as in training, not arbitrary photos.

Cost reports distinguish dense encoder MACs, dense classifier MACs and event
proxies. Runtime includes dense encoding/statistics. Fewer spikes are not
measured energy savings. The software checks do not establish full MNIST v3
accuracy, CIFAR-100 performance, or scientific superiority.
