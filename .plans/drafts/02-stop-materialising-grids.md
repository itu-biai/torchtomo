# 02. Stop materialising the rotation grids

## The problem

`ParallelBeam.__init__` precomputes and registers two full sampling grids:

```python
self.register_buffer("forward_grids", self._precompute_forward_grids(grid_x, grid_y))
self.register_buffer("backward_grids", self._precompute_backward_grids(grid_x, grid_y))
```

Each is `[n_angles, img_size, img_size, 2]` in float32. At 512 px and 360 angles
that is 755 MB each, **1.51 GB resident on the card before a single projection is
computed**, and it scales linearly with both angle count and pixel count.

This never appears in the per-call memory column of the benchmark, because it is
allocated at construction. It is still the largest single allocation the library
makes, and it is charged once per projector. Subset projectors multiply it:
`Noise2Inverse` with 3 splits builds three more, and `angle_subsets` for SART with
one view per subset builds 90.

What the buffer stores is a rotation of a fixed coordinate grid:

```python
x_rot = cos_a * grid_x + sin_a * grid_y
y_rot = -sin_a * grid_x + cos_a * grid_y
```

Two multiplies and an add per element, from a `[H, W]` base grid shared by every
angle. The precomputation trades 8 bytes per element of bandwidth for roughly
three flops, which on any modern GPU is the wrong side of the trade: the kernel is
memory bound either way, and reading a shared `[H, W]` base grid hits cache where
reading a private `[A, H, W, 2]` slab does not.

## The approach

Keep only the base coordinate grid, `[H, W, 2]` or two `[H, W]` buffers, and build
each chunk's sampling grid inside the loop:

```python
def _forward_grid(self, start, end):
    angles = self.angles[start:end].view(-1, 1, 1)
    cos_a, sin_a = angles.cos(), angles.sin()
    return torch.stack(
        (cos_a * self.grid_x + sin_a * self.grid_y, -sin_a * self.grid_x + cos_a * self.grid_y),
        dim=-1,
    )
```

This composes with draft 01: that draft wants a `[1, A_c * H, W, 2]` grid, which
this can produce directly with a `reshape`, so the two land together.

`backward_grids` gets the same treatment. Its construction differs (detector
lookup rather than image rotation) so read `_precompute_backward_grids` and mirror
it exactly rather than assuming symmetry.

Keep the precomputed path behind a constructor flag, `cache_grids=False` by
default, for anyone who is running a fixed small geometry in a loop and would
rather spend the memory. Do not make it the default: the default should be the one
that does not fall over at 720 angles.

## Expected gain

Frees 1.51 GB at 512 px and 360 angles, proportionally more at larger geometries.
Likely also a small speedup from better cache behaviour, but treat that as a bonus
and measure it rather than promising it.

The subset case is where this matters most. SART with 90 one-view subsets
currently constructs 90 projectors; if each precomputes grids for its own single
angle the total is unchanged, but `Noise2Inverse` with 3 splits at full size is
paying 4.5 GB of grids today.

## Risk

Low for correctness, the expression is copied from the existing precompute.

The real risk is a per-call slowdown if the grid construction is not fused into
the sampling. Measure the chunk loop before and after; if constructing the grid
costs more than the 0.630 ms copy it replaces, the change is not worth it on its
own, though it still pays for itself in memory.

## Validation

1. `pytest -q`, 69 passed.
2. Assert `_forward_grid(0, n_angles)` equals the old `forward_grids` buffer
   exactly, and likewise for backward, in float64. This is the whole correctness
   argument and it is a two line test.
3. Confirm the resident memory drop with `torch.cuda.memory_allocated()` around
   projector construction at 512 px and 360 angles.
4. `benchmark/test_leap_consistency.py`.
