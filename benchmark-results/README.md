# torchtomo against LEAP and torch-radon

Measured on one RTX 2080 Ti, torch 2.4.0+cu121, with
`benchmark/compare_libraries.py`. Each library projects the same phantom and
reconstructs its own sinogram, so the scale each one works in cancels and nothing
is corrected by a fitted factor. All three get the same angle list, the same
phantom, and the same inscribed circle to be scored over.

Raw numbers are in `library-comparison.json`. `library-comparison.png` shows the
reconstructions and their error maps, `library-summary.png` the four dimensions
side by side.

## The operators agree

Sinograms match to between 0.05% and 0.2% in relative L2 across every size, angle
count and phantom tested. The scale each library needs to reach torchtomo's is
exact and explainable:

| Library | Scale to torchtomo | Why |
| --- | --- | --- |
| LEAP | 1.000000 | same convention once the arc is negated |
| torch-radon | `2 / size` exactly | it sums pixel values where torchtomo integrates over a pixel width |

This is a stronger check than the library had before: the torch-radon comparison
could only match up to a fitted ratio, while LEAP confirms the absolute scale.

## Reconstruction quality, Shepp-Logan

PSNR in dB and SSIM, over the visible circle. Measured with the Ram-Lak ramp that
landed in 0.3; the row below the table is what the same runs gave before it.

| Geometry | torchtomo | torchtomo-cuda | LEAP | torch-radon |
| --- | --- | --- | --- | --- |
| 256 px, 90 angles | 25.59 / 0.648 | 25.59 / 0.648 | 25.55 / 0.618 | 25.59 / 0.648 |
| 256 px, 180 angles | 27.07 / 0.850 | 27.07 / 0.850 | 27.33 / 0.819 | 27.07 / 0.850 |
| 256 px, 360 angles | 27.46 / 0.937 | 27.46 / 0.937 | 27.89 / 0.923 | 27.46 / 0.937 |
| 512 px, 90 angles | 24.48 / 0.534 | 24.48 / 0.534 | 24.20 / 0.511 | 24.48 / 0.534 |
| 512 px, 180 angles | 28.48 / 0.745 | 28.48 / 0.745 | 28.39 / 0.714 | 28.48 / 0.745 |
| 512 px, 360 angles | 30.08 / 0.909 | 30.08 / 0.909 | 30.35 / 0.886 | 30.08 / 0.909 |

torchtomo and torch-radon now agree to two decimals in both metrics, which is what
two libraries applying the same discrete ramp to the same measurements should do.
Against LEAP it leads on SSIM in every row and on PSNR at 90 views, and trails by
0.26 to 0.43 dB at 180 and 360 views. The `cuda` backend reproduces the PyTorch
path to the same two decimals; its own kernels are the only difference.

Before the ramp fix the same table read 24.27 / 0.539 at 256 px and 90 views up to
28.78 / 0.682 at 512 px and 360 views, last in every row and by 0.2 of SSIM at the
bottom of it. Both causes were in the FBP filter rather than in the projector, and
both were the same one bin.

## The DC bin was the whole story

The ramp was built analytically as `freq.abs()`, so the DC bin of the padded
transform was exactly zero and the mean of every projection was discarded. The
reconstruction then sat a constant -0.0172 below the phantom, where LEAP's and
torch-radon's biases were zero to five decimal places: the error maps in the
`library-comparison.png` of that time show torchtomo's interior uniformly blue and
the other two white.

torch-radon and skimage instead build Kak and Slaney's kernel (Chapter 3,
Equation 61) in the spatial domain and transform it, which leaves a small positive
DC term, 2 / (pi^2 M) for a transform of length M. Normalised by their maxima the
two filters agree bin for bin to within 3% and differ most at that one bin,
0.000396 against 0.

`src/torchtomo/filters.py` now builds the ramp the second way, for every window on
top of it. The reconstruction bias at 512 px and 360 views is +0.00004, and the
quality table above is the rest of the effect.

## FBP inverts its own forward projector

Reprojecting a reconstruction should return the measurements it came from. That
test used to fail in a way a constant could not fix: torchtomo plateaued at a 10%
residual whatever the angle count, two thirds of it a fixed 9.4% amplitude deficit,
while LEAP converged to 0.45%. Dividing by the fitted gain halved the residual and
cost PSNR, which said the deficit was not a scale.

It was the missing DC bin, which carries about a third of a sinogram's energy. With
the ramp built from the spatial kernel, gain and residual land on LEAP's, from
`benchmark/fbp_consistency.py`, 512 px, Shepp-Logan, gain / residual / bias:

| Angles | torchtomo | LEAP |
| ---: | ---: | ---: |
| 45 | 1.0018 / 0.0611 / +0.00001 | 1.0018 / 0.0638 / +0.00001 |
| 90 | 1.0000 / 0.0173 / +0.00001 | 1.0001 / 0.0187 / +0.00000 |
| 180 | 1.0002 / 0.0054 / +0.00007 | 0.9998 / 0.0056 / -0.00000 |
| 360 | 0.9999 / 0.0049 / +0.00004 | 0.9997 / 0.0045 / -0.00000 |
| 720 | 0.9998 / 0.0051 / +0.00002 | 0.9997 / 0.0048 / +0.00000 |

Residual is `||A fbp(y) - y|| / ||y||` and gain is the least squares scale between
the two. torchtomo is ahead at 45 and 90 views and level from 180 on. This is what
any method whose loss round-trips through `fbp()` and `forward()` was paying for:
BM3D, RED, Noise2Inverse and Proj2Proj in `examples/ellipses`.

Fan beam had the same defect and less of it, because a detector 1.5x wider than the
image leaves less of the projection mean in the DC bin: gain 0.9753 -> 0.9991,
residual 0.0263 -> 0.0049, bias -0.00446 -> -0.00010 at 512 px and 360 views.


## Speed

Milliseconds per call and images per second, 512 px, batch 4. The torchtomo
column is after the projector rewrite described under "What the profile said"
below; the numbers it replaced are in that section.

| Operation | Angles | torchtomo | LEAP | torch-radon |
| --- | --- | --- | --- | --- |
| forward | 90 | 3.287 ms, 1217/s | 1.129 ms, 3542/s | 0.227 ms, 17602/s |
| backproject | 90 | 12.426 ms, 322/s | 0.620 ms, 6455/s | 0.163 ms, 24538/s |
| fbp | 90 | 3.093 ms, 1293/s | 3.486 ms, 1147/s | 0.220 ms, 18202/s |
| forward | 360 | 17.879 ms, 224/s | 2.599 ms, 1539/s | 0.674 ms, 5936/s |
| backproject | 360 | 54.485 ms, 73/s | 1.661 ms, 2408/s | 0.630 ms, 6345/s |
| fbp | 360 | 14.284 ms, 280/s | 7.659 ms, 522/s | 0.883 ms, 4532/s |

torch-radon is fastest throughout, by 7x to 86x over torchtomo on projection and
backprojection. LEAP sits between the two, 2.9x to 33x ahead. The gap is widest on
backprojection, where torchtomo pays for `adjoint()` differentiating a temporary
graph rather than running a transpose kernel.

torchtomo wins FBP at batch 1 up to 512 px with 90 angles (1.76 ms against LEAP's
2.96 ms), because LEAP carries a fixed per-call overhead of a few milliseconds
that does not shrink with the problem; part of that is the copy the adapter in
`benchmark/leap_projector.py` makes before handing the sinogram over. At 512 px
with 360 angles that overhead is amortised and LEAP takes the row, 5.20 ms against
8.56 ms.

## GPU memory

Peak for a single call, 512 px, 360 angles, batch 4.

| Operation | torchtomo | LEAP | torch-radon |
| --- | ---: | ---: | ---: |
| forward | 174.7 MB | 10.9 MB | 2.9 MB |
| backproject | 615.1 MB | 8.4 MB | 4.2 MB |
| fbp | 174.9 MB | 10.9 MB | 43.7 MB |

This is still the widest gap of the four dimensions, though it is 7.7x narrower on
backprojection than it was. torchtomo's `adjoint()` builds and differentiates a
temporary forward graph on every call, which is what keeps it at 615 MB where the
two CUDA libraries spend under 10 MB. Before the rewrite the same call cost
4709 MB, and that was the direct reason LPD at 512 with `--lpd-iterations 10
--lpd-width 32` ran out of memory above batch 2 on an 11 GB card. It now trains at
batch 5, peaking at 9.30 GB.

LEAP allocates outside PyTorch's caching allocator, so `max_memory_allocated`
cannot see it. Both a torch-level and a driver-level figure are recorded in the
JSON, and they agree closely here.

Separately, the projector itself holds memory before any call is made. That was
1511 MB of precomputed rotation grids at 512 px and 360 angles; it is now 271.6 MB
of on-demand grids under a cache budget, and 3.1 MB with the cache disabled.

## What the profile said

Most of the original gap was not the price of staying in pure PyTorch. At 512 px,
360 angles, batch 4, more time went to `expand().reshape()` copying tensors
(0.630 ms per chunk) than to `grid_sample` doing the work (0.485 ms). Removing
that duplication and building the sampling grids on demand gave:

| Operation | before | after | gain | peak before | peak after |
| --- | ---: | ---: | ---: | ---: | ---: |
| forward | 29.94 ms | 17.88 ms | 1.67x | 342.5 MB | 174.7 MB |
| backproject | 76.83 ms | 54.49 ms | 1.41x | 4709.1 MB | 615.1 MB |
| fbp | 23.58 ms | 14.28 ms | 1.65x | 275.7 MB | 174.9 MB |

Smaller geometries gain more, up to 2.44x on forward and 3.17x on FBP at 256 px
with 90 angles. Every output is unchanged: CPU results are bitwise identical and
CUDA differs by one ulp, `benchmark/benchmark_adjoint.py --pairs 500 --dtype
float64` gives a residual of 3.6e-16, and `benchmark/test_leap_consistency.py`
still pins the forward to 0.1% against LEAP.

Two things that looked promising did not work. Computing the discrete adjoint
directly as a bilinear scatter was exact but 3x slower than the VJP, because
`scatter_add_` serialises on atomics. Raising the angle chunk bound bought 5% of
forward time for 12x the peak memory.

The remaining gap is structural: a rotate-and-sum forward touches every pixel for
every angle, where a ray-driven CUDA kernel walks only the pixels a ray crosses.
Closing it means writing that kernel, which is the thing torchtomo exists not to
do. If you need LEAP's speed, `benchmark/leap_projector.py` is a drop-in
`ParallelBeam` and `--projector leap` routes the whole benchmark through it.

## What it costs the benchmark: ten methods on each backend

`examples/ellipses/train.py --projector leap` swaps the kernels and changes
nothing else, so the same ten methods, schedules and seeds run on LEAP's
operators. Real CT, 512 px, 90 angles, 100k photons, in-window PSNR:

| Method | torchtomo | LEAP | Difference |
| --- | ---: | ---: | ---: |
| FBP | 12.59 | 12.25 | -0.34 |
| SIRT | 19.98 | 19.96 | -0.02 |
| SART | 20.08 | 20.03 | -0.05 |
| iRadonMAP | 23.17 | 23.20 | +0.02 |
| FBP + U-Net | 25.47 | 25.89 | +0.42 |
| LPD | 22.52 | 23.47 | +0.94 |
| RED | 24.08 | 25.57 | +1.49 |
| Noise2Inverse | 18.47 | 20.86 | +2.39 |
| FBP + BM3D | 19.54 | 24.66 | +5.12 |
| Proj2Proj | 13.29 | 23.19 | +9.90 |
| noiseless FBP reference | 19.73 | 24.04 | +4.30 |

Ellipse phantoms at 512 px show the same ordering, with Proj2Proj gaining 12.95 dB
over the circle.

The pattern follows the operators rather than the methods. SIRT, SART and
iRadonMAP match within 0.05 dB: they never call `fbp()`, using only the matched
`forward()` and `adjoint()` pair, which the two libraries agree on. Everything
that routes through `fbp()` improves, and the method that improves most is the one
whose training loss round-trips through both: Proj2Proj perturbs a sinogram,
reconstructs it, denoises, forward projects the result, and compares with the
measurements. On torchtomo it has to absorb the 10% round-trip inconsistency
before it can denoise anything, from a masked loss that sees one sixteenth of the
entries. That is a hypothesis consistent with every row in the table, not a proven
mechanism, but the SIRT and SART rows make it hard to explain any other way.

## The FBP baseline is the one untrained method whose hyperparameter is never searched

`classical.py` selects SIRT's iteration count, SART's relaxation, BM3D's sigma and
RED's prior weight on the validation split, and records each search. FBP's filter
is not among them: `fbp()` is called with its default `"ramp"` everywhere. On noisy
data that is the sharpest and noisiest choice available, and it is worth several
decibels.

torchtomo's own filters on the real CT slices, selected on validation exactly as
the other untrained methods are:

| Filter | Val circle | Val window | Test circle | Test window |
| --- | ---: | ---: | ---: | ---: |
| ramp (current, fixed) | 23.470 | 12.905 | 23.715 | 12.585 |
| cosine | 27.877 | 15.863 | 28.322 | 15.481 |
| shepp-logan | 28.468 | 16.304 | 28.953 | 15.937 |
| hamming | 29.157 | 16.818 | 29.693 | 16.479 |
| **hann** | **29.509** | **17.087** | **30.080** | **16.775** |

Hann wins on validation and carries over to test. Selecting the filter the way
every other untrained method is selected moves the FBP row from **12.59 dB to
16.78 dB in window**, and from 23.72 to 30.08 over the circle. That reorders the
table: FBP stops being far below Proj2Proj and closes much of the distance to SIRT
and SART.

The same effect appears in LEAP, whose ramp filter order is also tunable. On the
same slices its order 0 filter reaches 17.85 dB in window against 12.26 for
Ram-Lak, a 5.6 dB spread from one parameter.

This is worth fixing before the table is published, for two reasons. The
comparison currently flatters every method that post-processes an FBP image,
because it hands them the worst available starting point, and it understates the
analytic baseline the learned methods are being measured against.

## Reproducing

```bash
# quality and figures, tolerant of a busy card
PYTHONPATH=src:benchmark python benchmark/compare_libraries.py \
    --output benchmark-results --sections quality figure summary

# speed and memory, which need the card to themselves
PYTHONPATH=src:benchmark python benchmark/compare_libraries.py \
    --output benchmark-results --sections performance
```
