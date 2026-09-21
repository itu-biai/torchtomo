# Benchmarks

Speed and self-consistency checks of torchtomo itself, on torch alone: nothing
here needs a dependency the library does not already have, nothing here is
needed to use `torchtomo`, and the library never imports any of it.

| File | What it does |
| --- | --- |
| `benchmark_speed.py` | forward, adjoint, and FBP time for every device and backend available |
| `benchmark_adjoint.py` | inner-product test of `forward()` against `adjoint()` |
| `calibrate_geometry.py` | recovers a scanner's axis offset by differentiating the geometry |

```bash
make benchmark-speed      # the backend speed table
make benchmark-adjoint    # 500 pairs in float64
make benchmark-calibrate  # a 3 px axis offset, recovered from the sinogram alone
```

`calibrate_geometry.py` knows the sinogram and nothing else: no phantom, no true
shift. It descends the reconstruction's own data consistency,
`|| A_u fbp_u(y) - y ||^2`, in the detector shift `u`, and the gradient comes
from the projector. Defaults to 256 px and 256 views on CUDA where there is one;
`--geometry both`, `--noise`, `--size` and `--device cpu` are the interesting
knobs.

Everything that needs another library to say anything lives in
[torchtomo-benchmark](https://github.com/itu-biai/torchtomo-benchmark): the
comparisons against scikit-image, LEAP, and torch-radon, the reconstruction
methods trained on top of torchtomo (FBP+U-Net, Learned Primal-Dual, and the
rest), and every recorded result.
