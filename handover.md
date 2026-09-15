# Handover

Written 2026-09-15. Everything below is in the working tree; **nothing is committed yet**.

## What this is

`torchtomo` is a pure-PyTorch differentiable CT library. The active work is the
benchmark in `examples/ellipses/`, which compares ten reconstruction methods on
synthetic ellipse phantoms and on real CT slices, using the library's matched
`forward()` / `backward()` operator pair.

The example deliberately adds **no dependency** beyond what the project already has
(PyTorch, NumPy, and Matplotlib from the dev extra). That constraint is why BM3D is
implemented from scratch rather than imported.

## Setup

Python 3.10 or newer, one CUDA card. Peak memory measured at 512 x 512 with batch 5
was under 4 GB for every method, so a 40 GB card is ample and the batch size can go up.

```bash
pip install -e ".[dev]"          # torch, numpy, matplotlib, pytest, ruff
export PYTHONPATH=src:examples/ellipses
pytest -q                        # expect 69 passed, 2 skipped
```

Copy `examples/ellipses/ct-subset/` across with the repository. It is the packed CT
data, 70 MB, gitignored, and is what the CT run reads. Only regenerate it if you need a
different subset:

```bash
python examples/ellipses/pack_ct_subset.py \
    --source /path/to/dataset_biailab_I2I_2025v1 \
    --output examples/ellipses/ct-subset --train 200 --val 50 --test 50
```

The raw 5.4 GB dataset it was built from is at
`~/projects/itu-biai/dataset/dataset_biailab_I2I_2025v1` on the old machine.

## Immediate task: run the two 512 x 512 experiments

Both were attempted and neither produced output, so these numbers do not exist yet.
They are the first thing to run.

```bash
# 1. Real CT slices, ten methods         (~45 min on one A100)
python examples/ellipses/train.py \
    --data-dir examples/ellipses/ct-subset --photons 100000 \
    --image-size 512 --angles 90 --batch-size 5 --device cuda \
    --lpd-iterations 10 --lpd-width 32 --unet-epochs 60 --lpd-epochs 60 \
    --sirt-iterations 300 --output examples/ellipses/results-ctw

# 2. Ellipse phantoms, ten methods       (~30 min on one A100)
python examples/ellipses/train.py \
    --image-size 512 --angles 90 --batch-size 5 --device cuda \
    --lpd-iterations 10 --lpd-width 32 --sirt-iterations 300 \
    --output examples/ellipses/results-512-all
```

Both write a full set of figures, `metrics.json`, `classical-selection.json`,
`training-summary.json`, and a `<model>-history.json` per network. Long runs are
interruptible: every epoch writes `<model>-last.pt`, and `--resume` continues on the
same schedule.

## The ten methods

| Name | Trained on | What it is |
| --- | --- | --- |
| `fbp` | nothing | ramp-filtered backprojection |
| `sirt` | nothing | simultaneous algebraic reconstruction, stopped early |
| `sart` | nothing | same update one view subset at a time |
| `bm3d` | nothing | collaborative filtering of the FBP image |
| `fbp-unet` | clean images | residual U-Net on the FBP image |
| `red` | borrows the U-Net | Regularization by Denoising with that U-Net as prior |
| `iradonmap` | clean images | learnable filtering + backprojection, then refinement |
| `noise2inverse` | measurements only | learns between reconstructions of disjoint view subsets |
| `proj2proj` | measurements only | J-invariant masking in the projection domain |
| `lpd` | clean images | unrolled Learned Primal-Dual |

`--models` selects which networks to train; `--skip-classical` drops the untrained
companions. Both make partial runs cheap.

## Layout

| File | Role |
| --- | --- |
| `train.py` | the driver: data, training loop, evaluation, figures |
| `data.py` | phantoms, real CT loading, Poisson noise, PSNR, display windows |
| `models.py` | `FBPUNet`, `LearnedPrimalDual`, `IRadonMap`, operator norm |
| `objectives.py` | `Supervised`, `Noise2Inverse`, `Proj2Proj` training objectives |
| `classical.py` | SIRT, SART, RED, BM3D wiring, and the validation search |
| `bm3d.py` | BM3D in pure PyTorch |
| `pack_ct_subset.py` | turns the raw CT dataset into three compact archives |
| `profile_models.py` | per-batch timing (do not rename back to `profile.py`) |
| `colab_*.py` | legacy drivers for the machine this came from, safe to delete |

## How the training loop is organised

`train_model` is objective-driven. An objective supplies three things:

- `loss(model, ids, device, step)` for one batch, where `step` is a global counter so
  methods that cycle a mask or a held-out subset keep advancing across epochs
- `validation(model, ids, device, batch_size)`, the value checkpoints are selected on
- `reconstruct(model, ids, device, batch_size)` for evaluation

`uses_truth` says whether the objective touches the ground truth. **The self-supervised
methods select on their own measurement-domain loss**, never on PSNR against the clean
image. `val_psnr_db` is recorded in the history for plotting only. Keep it that way, or
their numbers stop meaning anything.

Adding a method means adding a branch in `build()` inside `train.py:main`, a label in
`save_plots`, and an entry in the `order` tuple.

## Results so far

64 x 64 phantoms, 100 images, 60/20/20 split, test-set PSNR over the visible circle:

| Method | Trained on | PSNR | Selected |
| --- | --- | ---: | --- |
| FBP | nothing | 21.74 dB | n/a |
| SIRT | nothing | 26.98 dB | 28 iterations |
| SART | nothing | 27.02 dB | 269 updates, relaxation 0.1 |
| FBP + BM3D | nothing | 28.32 dB | sigma 0.1 |
| iRadonMAP | clean images | 28.47 dB | epoch 118/120, lr 1e-4 |
| Noise2Inverse | measurements only | 28.48 dB | epoch 27/120 |
| Proj2Proj | measurements only | 25.94 dB | epoch 112/120 |
| FBP + U-Net | clean images | 30.26 dB | epoch 26/120 |
| RED | borrows the U-Net | 30.22 dB | 1 iteration, weight 10 |
| LPD | clean images | 30.15 dB | epoch 120/120 |

Noiseless FBP reaches 32.39 dB on the same split, recorded as
`noiseless_fbp_reference` in `metrics.json`.

Real CT, 512 x 512, 90 angles, 100k photons, **six methods only** (`results-ctw/`,
predates the four added since), in-window PSNR: FBP 12.59, SIRT 19.98, SART 20.08,
FBP+U-Net 25.46, RED 24.14, LPD 23.40. Noiseless FBP in-window is 19.73 dB.

## Open questions, most important first

### 1. LPD is under-trained, not broken

It selects the **final** epoch in nearly every run and its validation loss sits at
1.1x its training loss, so it is nowhere near overfitting. The published method trains
far longer than the 1,440 to 2,400 optimiser steps these schedules give it.

A 500 epoch diagnostic at 64 x 64 supports this. By epoch 105 LPD had already reached
**31.04 dB** validation against the 30.80 dB it finished the 120 epoch schedule with,
and against FBP+U-Net's 31.12 dB, with 395 epochs still to run. Finish or repeat it:

```bash
python examples/ellipses/train.py --output /tmp/diag-lpd500 \
    --models lpd --skip-classical --lpd-epochs 500
```

If it passes FBP+U-Net, give LPD a longer schedule than the other networks in the 512
runs and say so in the README, because a shared schedule is then no longer the fair
comparison it was meant to be. The operator path itself is verified: the adjoint passes
a matrix transpose test, `gradcheck`, and `gradgradcheck`, gradients reach the first
primal and dual blocks (asserted every run), and LPD reconstructs from a **zero** image
without ever seeing an FBP, which it could not do with a broken adjoint.

### 2. iRadonMAP: diagnosed and fixed, but check it holds at 512

On the shared 1e-3 schedule its validation MSE was **253x** its training MSE. The cause
was the learning rate, not the architecture: its back-projection weights start at
`angle_step`, about 0.035 at 90 angles, so one Adam step of 1e-3 moves each by a few
percent of its own value and pulls apart the analytic reconstruction the layer is
initialised to. Measured at 64 x 64:

| Learning rate | Test PSNR | Best epoch | val/train |
| --- | ---: | --- | ---: |
| 1e-3, the shared schedule | 27.23 dB | 24/120 | 253 |
| **1e-4, now the default** | **28.47 dB** | 118/120 | 2.8 |
| 2e-5, the paper's RMSProp rate | 24.07 dB | 120/120 | 1.4 |

`--iradon-learning-rate` now defaults to 1e-4. At that rate it selects epoch 118 of 120,
so it is still improving and the 28.47 dB is a floor. **Confirm the 1e-4 choice still
holds at 512 x 512**, where `angle_step` is unchanged but the layer has 23.6 M weights
instead of 369 k.

The implementation itself is verified: the untrained network reproduces `fbp()` to
3e-6, and gradients reach both learnable layers.

### 3. Proj2Proj is budget-limited

Its loss uses one sixteenth of the sinogram entries per step, so it extracts far less
signal per step than the others. The paper uses 200,000 iterations and a 2.16 M
parameter five-scale U-Net; here it gets ~1,440 steps and a 118,305 parameter two-level
one. It selects epoch 112 of 120, still improving. Give it its own longer schedule
before reading anything into its number.

### 4. BM3D is a lower bound

Validated against the reference `bm3d` package on white Gaussian noise with sigma tuned
for both: this implementation reaches 32.2 dB where the reference reaches 33.8 dB. The
gap is most likely the reference profile's bior1.5 block transform in the hard
thresholding stage, where this uses a DCT; swapping in a Haar transform recovers 0.4 dB
of it. Implementing bior1.5 would close most of the rest if the row matters.

### 5. Housekeeping

- Nothing is committed. `git status` shows the full set of new files.
- Result PNGs total roughly 10 MB across `results*/`. Panels are already capped at
  256 px by `--grid-tile`; pass `0` for full resolution.
- `results-512/` and `results-512-lpd10/` are the older LPD capacity ablation and
  cover three methods only. They are kept deliberately; do not overwrite them.

## Conventions to preserve

- **No new dependencies** in the project or the example.
- **No em dashes or en dashes** in any document (user's standing instruction).
- Self-documenting code; comment only genuinely non-obvious logic.
- Hyperparameters are chosen on the **validation** split, never the test split. Every
  untrained method records its search in `classical-selection.json`.
- Metrics are reported over the full image, the visible circle, and the display window
  where one exists. Quote the circle or window figure, not the full square, which the
  masked corners inflate by about a decibel.

## Gotchas

- `training.log` and `console.log` are **append mode**. Do not wait on a completion
  marker without recording a byte offset first; a stale marker from an earlier run will
  match immediately. This bit twice.
- `profile_models.py` must not be renamed back to `profile.py`; it shadows the stdlib
  `profile` module that `cProfile` imports, which breaks `torch.optim`.
- Subset projectors rescale `angle_step` so a subset `fbp()` keeps the parent's
  normalisation. Noise2Inverse depends on this; without it its input and target sit at
  different brightnesses.
- `dataset.pt` is roughly 800 MB for the 300 slice CT run at 512. It is gitignored and
  rebuilt by any non-resumed run.
- `--evaluate-only` reloads `dataset.pt` and `<model>-best.pt` and regenerates metrics
  and figures without retraining. It needs those files, so download them from the
  cluster if you want to redo figures locally.
- `--resume` restores weights, optimiser, scheduler, shuffle stream, history and epoch
  count, so interrupted long runs continue on the same schedule.

## What to keep from a run

The `.pt` files are gitignored on purpose and are rebuilt by any non-resumed run. Keep
`metrics.json`, `classical-selection.json`, `training-summary.json`, the
`*-history.json` files, and the PNGs; those are the record.

Keep `dataset.pt` and `<model>-best.pt` on the server if you want to redo figures or
metrics later without retraining, which is what `--evaluate-only` does.
