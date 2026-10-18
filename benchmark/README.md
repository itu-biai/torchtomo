# Benchmarks

Speed and self-consistency checks of torchtomo itself use the library's own
dependencies. The measured CT-PD pilot additionally uses the optional DICOM
reader and matplotlib. These scripts are separate from the installed library.

| File | What it does |
| --- | --- |
| `benchmark_speed.py` | forward, adjoint, and FBP time for every device and backend available |
| `benchmark_adjoint.py` | inner-product test of `forward()` against `adjoint()` |
| `calibrate_geometry.py` | recovers a scanner's axis offset by differentiating the geometry |
| `angles/learn_angles.py` | learns which views to measure by differentiating the angle column |
| `conebeam/check_feasibility.py` | standalone synthetic 3D helical reference, with a cylindrical detector, matched adjoint, geometry gradients, and CGLS |

```bash
make benchmark-speed      # the backend speed table
make benchmark-adjoint    # 500 pairs in float64
make benchmark-calibrate  # a 3 px axis offset, recovered from the sinogram alone
make benchmark-angles     # 12 views learnt for a class of objects, against evenly spaced ones
```

The cone beam feasibility script is an experiment separate from the public 2D
projectors. It integrates trilinearly sampled volumes along rays on a helical
trajectory, uses the same interpolation weights in its transpose, and generates
ray chunks on demand. Geometry uses millimetres. Its checks cover the explicit
matrix transpose, input gradients of both operators, finite differences of a
lateral detector shift, projections of an analytic Gaussian, and a synthetic
32 x 32 x 32 CGLS reconstruction. It writes `results.json` and
`synthetic_reconstruction.pt` to the output directory.

```bash
python benchmark/conebeam/check_feasibility.py \
    --device cuda --output benchmark/conebeam/results/local

colab run --gpu T4 -s torchtomo-helical-check --timeout 180 \
    benchmark/conebeam/check_feasibility.py
```

`colab run` releases its VM after execution; the JSON metrics are printed to
stdout. To retrieve the volume tensors as well, use a named `colab new` session,
run the script, download both output files, then run `colab stop`. Pass
`--device cpu` to the Python command for a local reference check. The script
requires only PyTorch. It does not load TCIA data, validate scanner corrections,
provide a public 3D API, or measure production CUDA kernels. The reconstruction
uses synthetic data from the same discrete operator; its residual measures
self-consistency rather than clinical reconstruction accuracy.

The measured-data pilot uses `conebeam/prepare_tcia.py` to decode one rotation
of CT-PD data into physical source and cylindrical-detector coordinates, and
`conebeam/reconstruct_tcia.py` to run the public experimental `ConeBeam` operator.
The preparation requires the optional `tcia` extra; the reconstruction figure
also requires matplotlib. Input directories are `full_proj/` and `full_img/`,
containing complete `.dcm` files.

```bash
python benchmark/conebeam/prepare_tcia.py \
    --root /path/to/L213 --output benchmark/conebeam/results/tcia
python benchmark/conebeam/reconstruct_tcia.py \
    --input benchmark/conebeam/results/tcia/input.npz \
    --output benchmark/conebeam/results/tcia/reconstruction
```

The pilot samples every fourth detector element and every fourth view from one
rotation. Alternating selected views are held out from reconstruction. Eight
training views and the vendor CT select a planar coordinate transform from the
manual's Cartesian frame to the DICOM patient frame; this step uses the reference
images and is reported explicitly. CGLS starts at zero. The output includes
training and held-out projection errors, a vendor comparison, volume arrays,
figures, and the selected frame. It is a coarse, single-case experimental check,
with no fan beam rebinning or statistical reconstruction weighting.

The 2026-10-18 full-dose GE L213 pilot used a `32 x 128 x 128` integration box
and 123 training plus 123 held-out views. Twenty CGLS iterations on a Colab T4
took 67.97 seconds including validation, with 357.74 MiB peak PyTorch allocation.
Final projection errors were 2.318% on training views and 3.044% on held-out
views. Only the central 16 mm slab was evaluated against the vendor CT; its
body correlation was 0.91244. The axial margins were partly observed, and image
agreement peaked before the final iteration while projection errors continued
falling. Regularization and broader scanner validation remain necessary.

The reader follows the
[TCIA DICOM-CT-PD manual, version 3](https://www.cancerimagingarchive.net/wp-content/uploads/DICOM-CT-PD-User-Manual_Version-3.pdf).
Data come from the
[LDCT-and-Projection-data collection](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/).
Its documentation requests acknowledgement of NIBIB grants EB017095 and
EB017185, Cynthia McCollough, PI.

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

Cross-library comparisons and learned reconstruction experiments live in
[torchtomo-benchmark](https://github.com/itu-biai/torchtomo-benchmark): the
comparisons against scikit-image, LEAP, and torch-radon, the geometry
calibration on measured scans against classical methods and the geometry-gradient
cost against Thies et al., the reconstruction
methods trained on top of torchtomo (FBP+U-Net, Learned Primal-Dual, and the
rest), and every recorded result.
