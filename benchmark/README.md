# Benchmarks

Speed and self-consistency checks of torchtomo itself, on torch alone: nothing
here needs a dependency the library does not already have, nothing here is
needed to use `torchtomo`, and the library never imports any of it.

| File | What it does |
| --- | --- |
| `benchmark_speed.py` | forward, adjoint, and FBP time for every device and backend available |
| `benchmark_adjoint.py` | inner-product test of `forward()` against `adjoint()` |
| `calibrate_geometry.py` | recovers a scanner's axis offset by differentiating the geometry |
| `angles/learn_angles.py` | learns which views to measure by differentiating the angle column |

```bash
make benchmark-speed      # the backend speed table
make benchmark-adjoint    # 500 pairs in float64
make benchmark-calibrate  # a 3 px axis offset, recovered from the sinogram alone
make benchmark-angles     # 12 views learnt for a class of objects, against evenly spaced ones
```

`calibrate_geometry.py` knows the sinogram and nothing else: no phantom, no true
shift. It descends the reconstruction's own data consistency,
`|| A_u fbp_u(y) - y ||^2`, in the detector shift `u`, and the gradient comes
from the projector. Defaults to 256 px and 256 views on CUDA where there is one;
`--geometry both`, `--noise`, `--size` and `--device cpu` are the interesting
knobs.

`angles/learn_angles.py` learns a 12-view angle set by descending the error of an
unrolled projected Landweber reconstruction over random ellipse phantoms, with
the gradient reaching the angles through every forward and adjoint. Objects
whose edges share an orientation gain from it; isotropic ones are the control
and should not. On an RTX 2080 Ti, 128 px, 30 iterations, 64 test phantoms:

| Objects | Evenly spaced | Random | Learnt |
| --- | --- | --- | --- |
| fibres, orientation std 15 degrees | 21.74 dB | 19.60 dB | 23.55 dB |
| isotropic | 21.69 dB | 19.27 dB | 21.68 dB |

Everything that needs another library to say anything lives in
[torchtomo-benchmark](https://github.com/itu-biai/torchtomo-benchmark): the
comparisons against scikit-image, LEAP, and torch-radon, the geometry
calibration on measured scans against classical methods and the geometry-gradient
cost against Thies et al., the reconstruction
methods trained on top of torchtomo (FBP+U-Net, Learned Primal-Dual, and the
rest), and every recorded result.
