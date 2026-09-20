# Benchmarks

Speed and self-consistency checks of torchtomo itself, on torch alone: nothing
here needs a dependency the library does not already have, nothing here is
needed to use `torchtomo`, and the library never imports any of it.

| File | What it does |
| --- | --- |
| `benchmark_speed.py` | forward, adjoint, and FBP time for every device and backend available |
| `benchmark_adjoint.py` | inner-product test of `forward()` against `adjoint()` |

```bash
make benchmark-speed      # the backend speed table
make benchmark-adjoint    # 500 pairs in float64
```

Everything that needs another library to say anything lives in
[torchtomo-benchmark](https://github.com/itu-biai/torchtomo-benchmark): the
comparisons against scikit-image, LEAP, and torch-radon, the reconstruction
methods trained on top of torchtomo (FBP+U-Net, Learned Primal-Dual, and the
rest), and every recorded result.
