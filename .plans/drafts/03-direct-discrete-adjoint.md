# 03. Compute the discrete adjoint directly instead of differentiating a graph

## The problem

`_DiscreteAdjoint.forward` in `src/torchtomo/base.py` gets `A^T y` by building a
throwaway forward graph and asking autograd for the vector-Jacobian product:

```python
with torch.inference_mode(False), torch.enable_grad():
    image = torch.zeros(..., requires_grad=True)
    projection = projector.forward(image)
    return torch.autograd.grad(projection, image, sinogram, create_graph=False)[0]
```

It is correct by construction, which is why it was written this way, and the
docstring is honest that it costs more than `backproject()`. The cost is larger
than it looks:

| | torchtomo | LEAP | torch-radon |
| --- | ---: | ---: | ---: |
| adjoint, 512 px, 360 angles, batch 4 | 105 ms | 1.7 ms | 0.64 ms |
| peak memory, same call | 4709 MB | 8.4 MB | 4.2 MB |

105 ms is 3.5x the 29.8 ms forward: one forward pass to build the graph, one
backward pass through `grid_sample`, plus graph bookkeeping. The 4.7 GB is the
retained activations.

This is the single most consequential inefficiency in the library. It is why LPD
at 512 with `--lpd-iterations 10 --lpd-width 32` cannot exceed batch 2 on an 11 GB
card, and therefore why LPD costs 44 minutes in the benchmark against 3 for the
U-Net. An implementation detail is deciding which methods look practical.

## The approach

The forward is a composition of two linear maps, so its transpose is the
composition of their transposes in reverse:

```
forward:  x  -> mask -> gather(bilinear, rotation grid) -> sum over rows -> * pixel_size
adjoint:  y  -> * pixel_size -> broadcast over rows -> scatter(bilinear) -> mask
```

Every piece has an easy explicit transpose:

- `* pixel_size` is its own transpose.
- the transpose of `sum(dim=2)` is `expand` along that dimension, free.
- the transpose of the circle mask is the same mask.
- the transpose of bilinear gather is bilinear scatter, which is `index_add_` with
  the same four corner indices and the same weights.

`_gather_bilinear` in `src/torchtomo/_sampling.py` already computes exactly those
indices and weights for the MPS path. The adjoint is the same function with
`flat_image.gather(2, index)` replaced by `result.index_add_(2, index, values)`.
Write `_scatter_bilinear` next to it, sharing the index and weight computation so
the two cannot drift apart.

Note the boundary handling: `_gather_bilinear` masks out-of-range samples with
`valid` and clamps the index. The scatter must apply `valid` to the weights before
accumulating, or clamped indices will deposit mass on the border. This is the one
place a transpose can silently stop being a transpose.

Keep `_DiscreteAdjoint` as the reference implementation and test the new path
against it, rather than deleting it.

## Expected gain

Speed: one scatter pass instead of a forward plus a backward, so 2x to 3x. It will
not approach LEAP's 1.7 ms, because `index_add_` on GPU serialises on atomics
where a ray-driven CUDA kernel accumulates in registers. Getting to 30 to 50 ms is
the realistic target.

Memory: no graph, no retained activations. The working set becomes the sinogram
plus the image plus the index tensors, tens of MB rather than 4.7 GB. That is the
part that matters, because it is what caps LPD's batch size.

## Risk

Medium, and higher than drafts 01 and 02. The transpose property is the thing the
whole library's LPD support rests on, and a subtly wrong scatter will still look
plausible: reconstructions will be nearly right and gradients slightly off, which
is the worst failure mode to debug.

Do not merge this on a PSNR comparison. Merge it on the inner-product test.

Atomic contention is the other unknown. Many rays deposit into the same voxel, and
if `index_add_` turns out slower than the VJP, the memory win alone may still
justify it, but that should be a deliberate decision rather than a surprise.

## Validation

In this order, and all of them before merging:

1. `benchmark/benchmark_adjoint.py --pairs 500 --dtype float64`. The inner product
   `<A x, y>` against `<x, A^T y>` is the definition. Anything above float64 round
   off means the scatter is wrong.
2. `torch.autograd.gradcheck` and `gradgradcheck`, which the handover records as
   passing today and which must keep passing.
3. Assert the new adjoint matches `_DiscreteAdjoint` to float64 round off on
   random sinograms, several sizes and angle counts, including a geometry where
   rays leave the image so the `valid` masking is exercised.
4. `pytest -q`, 69 passed.
5. Retrain LPD at 64 px for a short schedule and confirm the loss curve tracks the
   current one. The handover's assertion that gradients reach the first primal and
   dual blocks is checked every run and will catch a dead path.
6. Then re-measure batch scaling at 512 px to find the new LPD ceiling.
