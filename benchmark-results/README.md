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

PSNR in dB and SSIM, over the visible circle.

| Geometry | torchtomo | LEAP | torch-radon |
| --- | --- | --- | --- |
| 256 px, 90 angles | 24.27 / 0.539 | 25.57 / 0.618 | 24.63 / 0.608 |
| 256 px, 180 angles | 26.20 / 0.657 | 27.32 / 0.822 | 26.77 / 0.810 |
| 256 px, 360 angles | 26.74 / 0.718 | 27.88 / 0.923 | 27.39 / 0.915 |
| 512 px, 90 angles | 23.14 / 0.474 | 24.17 / 0.512 | 23.41 / 0.514 |
| 512 px, 180 angles | 27.13 / 0.592 | 28.36 / 0.716 | 27.85 / 0.719 |
| 512 px, 360 angles | 28.78 / 0.682 | 30.31 / 0.886 | 29.87 / 0.884 |

torchtomo is last everywhere, and its SSIM lags much further than its PSNR. Two
separate causes account for that, both in its FBP rather than in its projector.

## torchtomo's FBP carries a DC offset

Its reconstruction sits a constant -0.0172 below the phantom, where LEAP's and
torch-radon's biases are zero to five decimal places. The error maps in
`library-comparison.png` show it plainly: torchtomo's interior is uniformly blue,
the other two are white.

The cause is in `src/torchtomo/filters.py`. The ramp is built analytically as
`freq.abs()`, so the DC bin is exactly zero and the mean of every projection is
discarded. torch-radon and skimage instead build the Kak and Slaney kernel
(Chapter 3, Equation 61) in the spatial domain and transform it, which leaves a
small nonzero DC term. Normalised by their maxima the two filters agree bin for
bin to within 1% and differ only there, 0.000396 against 0.

Setting that one bin recovers almost all of the gap:

| Geometry | current | DC bin corrected | torch-radon |
| --- | --- | --- | --- |
| 512 px, 90 angles | 23.14 / 0.474 | 23.41 / 0.514 | 23.41 / 0.514 |
| 512 px, 360 angles | 28.78 / 0.682 | 29.85 / 0.877 | 29.87 / 0.884 |
| 256 px, 180 angles | 26.20 / 0.657 | 26.76 / 0.809 | 26.77 / 0.810 |

A residual bias of +0.0024 remains against torch-radon's 0.00000, so the exact
value is worth deriving rather than lifting, but one bin is the whole story.

**This has not been applied.** The benchmark runs in `examples/ellipses` were
produced with the current filter, and changing it would make those tables
incomparable with each other.

One caveat on how far this carries. On the noisy real CT slices the correction
helps the circle metric by +0.22 dB but costs 0.18 dB in the display window, so
it is a clear win on noiseless and phantom data and roughly neutral once Poisson
noise dominates.

## torchtomo's FBP is not a consistent inverse of its own forward

A sharper problem than the DC bin, and a separate one. Re-projecting a
reconstruction should return the measurements it came from:

| Angles | torchtomo gain | torchtomo residual | LEAP gain | LEAP residual |
| ---: | ---: | ---: | ---: | ---: |
| 45 | 0.9135 | 0.1441 | 1.0021 | 0.0721 |
| 90 | 0.9076 | 0.1048 | 1.0001 | 0.0203 |
| 180 | 0.9064 | 0.1002 | 0.9998 | 0.0058 |
| 360 | 0.9061 | 0.0998 | 0.9998 | 0.0045 |
| 720 | 0.9058 | 0.0998 | 0.9997 | 0.0048 |

Residual is `||A fbp(y) - y|| / ||y||`, gain is the least squares scale between
the two. LEAP converges to a 0.45% residual. torchtomo plateaus at 10% and stops
improving with more angles, because roughly two thirds of it is a fixed 9.4%
amplitude deficit that does not depend on the angle count.

Dividing the reconstruction by that gain halves the residual, to 0.037 at 360
angles, but does not reach LEAP and costs 0.2 dB of PSNR against the phantom. So
a single constant is not the whole fix: the current scale is close to the best one
for PSNR, while being the wrong one for self-consistency. The two goals are
genuinely in tension here, and the remainder is spectral rather than a scale.

This matters for any method whose loss round-trips through `fbp()` and
`forward()`. The next section is what that costs.

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
forward time for 12x the peak memory. Both are written up in `.plans/drafts/`.

The remaining gap is structural: a rotate-and-sum forward touches every pixel for
every angle, where a ray-driven CUDA kernel walks only the pixels a ray crosses.
Closing it means writing that kernel, which is the thing torchtomo exists not to
do. If you need LEAP's speed, `benchmark/leap_projector.py` is a drop-in
`ParallelBeam` and `--projector leap` routes the whole benchmark through it.

LEAP allocates outside PyTorch's caching allocator, so `max_memory_allocated`
cannot see it. Both a torch-level and a driver-level figure are recorded in the
JSON, and they agree closely here.

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
