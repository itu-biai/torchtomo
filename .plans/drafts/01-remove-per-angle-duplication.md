# 01. Remove the per-angle tensor duplication in forward projection

## The problem

`ParallelBeam.forward` builds a rotated copy of the whole image for every angle,
and a private copy of the rotation grid for every image in the batch:

```python
grid = self.forward_grids[start:end]                        # [A_c, H, W, 2], a view
grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)
grid = grid.reshape(B * A_c, H, W, 2)                       # materialises B copies
batch = x.unsqueeze(1).expand(-1, A_c, -1, -1, -1)
batch = batch.reshape(B * A_c, 1, H, W)                     # materialises A_c copies
rotated = sample_bilinear(batch, grid)
projection = rotated.sum(dim=2) * self.pixel_size
```

`expand` returns a view, but `reshape` on that view cannot be done by restriding,
so both lines allocate and copy. At 512 px, batch 4, chunk 16 that is 134 MB of
grid and 67 MB of image per chunk, 23 chunks per call, purely to satisfy
`grid_sample`'s requirement that input and grid share a batch dimension.

Measured: 0.630 ms per chunk against 0.485 ms for the sampling itself. **The copy
costs more than the work.**

## The approach

`grid_sample` does not require one grid per angle. It maps `[N, C, H_in, W_in]`
and `[N, H_out, W_out, 2]` to `[N, C, H_out, W_out]`, and `H_out` is arbitrary. So
stack the angles along the output height instead of the batch:

```python
grid = self.forward_grids[start:end].reshape(1, A_c * H, W, 2).expand(B, -1, -1, -1)
rotated = sample_bilinear(x, grid)                          # [B, 1, A_c * H, W]
projection = rotated.view(B, 1, A_c, H, W).sum(dim=3) * self.pixel_size
```

The image is passed through untouched, with no copy at all. The grid is reshaped
from a contiguous buffer, which is free, and then expanded, which is a view.

Two things to verify while implementing, both cheap to check and both of which
decide how much of the saving is real:

1. whether `grid_sample` accepts a batch-expanded (stride 0) grid without copying
   it internally. If it does not, materialise the grid once per chunk outside the
   angle loop rather than per call, or drop to `expand` only when `B == 1`.
2. whether the reduction is better as `sum(dim=3)` on a 5D view or as a reshape to
   `[B, A_c, H, W]` first. Time both; the 0.121 ms baseline is the number to beat.

The same rewrite applies to `backproject`, which has the identical pattern with
`sino_rows` and `backward_grids`, and to `IRadonMap.sinusoidal_backprojection` in
`examples/ellipses/models.py`, which copied the idiom.

## Expected gain

Removing 51% of forward time is a 2x forward if the sampling cost is unchanged.
Conservatively 1.5x, since a larger single `grid_sample` call may schedule
differently than 16 smaller ones. Everything inherits it: `fbp`, `adjoint` (which
calls forward twice), SIRT, SART, RED, LPD training.

## Risk

Low. The arithmetic is identical, the same coordinates are sampled in the same
order, and no interpolation changes. The chunking logic stays as it is.

The one behavioural difference is peak memory: a single `[B, 1, A_c * H, W]`
output instead of `[B * A_c, 1, H, W]` is the same size, so `_angle_chunk_size`
does not need retuning, though draft 04 revisits it.

## Validation

1. `pytest -q`, expect 69 passed, 2 skipped.
2. Assert the new forward matches the old one exactly on random input, at several
   sizes and angle counts, in float64. This should be bitwise or within 1 ulp; if
   it is not, the grid reshape is wrong.
3. `benchmark/test_leap_consistency.py`, which pins the forward to 0.1% against
   LEAP.
4. Re-run `benchmark/compare_libraries.py --sections performance` and compare
   against the numbers recorded in `benchmark-results/library-comparison.json`.
