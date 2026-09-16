# 04. Cheap wins to measure after the obvious waste is gone

These are small, independent, and worth trying only once drafts 01 to 03 have
landed, because each is easier to attribute when the big copies are no longer
dominating the profile.

## Retune the angle chunk size

`BaseProjector._angle_chunk_size` caps a chunk at `1 << 24` work items on GPU,
which at 512 px, batch 4 gives 16 angles and therefore 23 chunks and 23 kernel
launches per call. That bound was chosen to keep the per-angle expansion in check.
Draft 01 removes the expansion, so the bound is protecting against something that
no longer exists.

Sweep the target from `1 << 24` to `1 << 27` and plot time against peak memory at
256 and 512 px, batch 1 and 4. Expect the sweet spot to move up substantially.
Keep whatever bound still fits comfortably on an 11 GB card with a model resident,
since that is the machine the benchmark runs on.

## torch.compile the projector

`forward` after draft 01 is a reshape, a `grid_sample` and a reduction. Inductor
can fuse the reduction into the sampling epilogue and remove the intermediate
`[B, 1, A_c * H, W]` tensor entirely, which is the largest remaining allocation in
the call.

Gate it behind a flag rather than compiling on import, since compilation costs
seconds on first call and the library is used interactively. `torch.compile` with
`dynamic=False` is right here because the geometry is fixed by construction, which
is exactly the case Inductor handles best.

Measure, do not assume. `grid_sample` is not always fusible and the win may be
zero.

## Let the sum accumulate in the sampling dtype

`rotated.sum(dim=2)` at 512 px sums 512 bilinear samples per ray in float32. That
is fine numerically, but check whether `sum(dim=..., dtype=torch.float32)` on a
half-precision sample would be acceptable for the forward operator alone. This is
only worth pursuing if someone wants throughput badly enough to accept a different
numerical answer, which most of this library's users will not. Listed for
completeness, not recommended.

## Do not pursue

**Fourier slice / NUFFT forward projection.** Asymptotically this is the real
answer, `O(N^2 log N + A N log N)` against the current `O(A N^2)`, and at 512 px
with 360 angles it could plausibly be an order of magnitude. It is still the wrong
change for this library: it would produce a different operator, break the 0.1%
agreement with LEAP and torch-radon that the benchmark just established, require a
gridding kernel with its own interpolation error, and invalidate every checkpoint
trained against the current adjoint. torchtomo's claim is a matched, verifiable,
portable operator pair. Trading that for speed gives up the reason to use it.

**Custom CUDA kernels.** Same argument, more bluntly. If the answer is a hand
written CUDA kernel then the answer is to call LEAP or torch-radon, both of which
this repo can now do: `benchmark/leap_projector.py` is a drop-in `ParallelBeam`
and `--projector leap` already routes the whole ellipses benchmark through it.
