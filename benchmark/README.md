# Benchmarks

Speed and self-consistency checks of torchtomo itself use the library's own
dependencies. The measured CT-PD and VICTRE pilots additionally use the optional DICOM
reader and matplotlib. These scripts are separate from the installed library.

| File | What it does |
| --- | --- |
| `benchmark_speed.py` | forward, adjoint, and FBP time for every device and backend available |
| `benchmark_adjoint.py` | inner-product test of `forward()` against `adjoint()` |
| `calibrate_geometry.py` | recovers a scanner's axis offset by differentiating the geometry |
| `angles/learn_angles.py` | learns which views to measure by differentiating the angle column |
| `conebeam/check_feasibility.py` | standalone synthetic 3D helical reference, with a cylindrical detector, matched adjoint, geometry gradients, and CGLS |
| `dbt/prepare_victre.py`, `dbt/reconstruct_victre.py` | one TCIA VICTRE breast tomosynthesis phantom: line integrals, FBP, OS-SART, held-out views, and wrong-geometry controls |
| `dbt/calibrate_victre.py` | fits the VICTRE arc to its own projections from a wrong start, alternating SIRT and geometry gradient steps |

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

## Breast tomosynthesis

`dbt/prepare_victre.py` downloads one phantom of the
[TCIA VICTRE](https://www.cancerimagingarchive.net/collection/victre/) trial,
its 25 projections and VICTRE's own reconstruction, and converts the projections
with `torchtomo.victre`, the BIAI Lab DBT 2026v1 conversion: pedestal
subtraction, a per-view air level from a ring around the breast, binning before
the log, and masked collimator edges. `dbt/reconstruct_victre.py` then builds
`victre_projector` on that dataset's grid and runs

- FBP: cosine weighting, a Hann-windowed ramp along tube travel, and the matched
  transpose, compared with VICTRE's FBP by Pearson correlation inside the breast;
- OS-SART with the dataset's ground-truth recipe (5 subsets, 10 iterations,
  nonnegative, 8 extra slices above and below), fitted to the 13 even views and
  scored on the 12 odd ones;
- both again with the source above the detector centre instead of the chest-wall
  edge, and with the view order reversed.

Both scripts need the `tcia` extra; the figure needs matplotlib.

```bash
python benchmark/dbt/prepare_victre.py --output benchmark/dbt/results/victre
python benchmark/dbt/reconstruct_victre.py \
    --input benchmark/dbt/results/victre/input_bin8.npz \
    --output benchmark/dbt/results/victre/bin8
```

The 2026-10-19 run used phantom 694607065 (scattered, 57 slices, in the
DBT 2026v1 training split) on a Colab T4 with PyTorch 2.11.0. Bin 8 is a
376 x 192 detector at 0.68 mm and a 57 x 168 x 204 volume; bin 4 is the
dataset's 752 x 384 detector at 0.34 mm and 57 x 336 x 408 volume.

| Bins | Geometry | FBP correlation | OS-SART correlation | Training residual | Held-out residual |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 | published | 0.886 | 0.448 | 3.99% | 3.58% |
| 8 | source over detector centre | 0.522 | 0.149 | 4.09% | 3.63% |
| 8 | reversed view order | -0.510 | -0.304 | 49.1% | 47.3% |
| 4 | published | 0.884 | 0.435 | 3.98% | 3.65% |
| 4 | source over detector centre | 0.483 | 0.139 | 4.07% | 3.69% |
| 4 | reversed view order | -0.382 | -0.283 | 48.8% | 47.0% |

At bin 4, FBP took 36 s and OS-SART 400 s, with 4.2 GiB peak PyTorch allocation;
at bin 8, 4.5 s and 51 s. On ConeBeam's CUDA kernels (`--backend cuda`, the
default `auto` on a GPU), an RTX 2080 Ti reproduces the bin 4 scores to three
decimals with FBP in 0.08 s and OS-SART in 0.37 s, in 736 MiB. The published geometry agrees with VICTRE's FBP better
than the lab's LEAP FBP did on three scattered phantoms (0.866) and less well
than its pyDBT FBP (0.920), whose placement was tuned on the reference; those
used their own mask and 0.34 mm binning. The OS-SART breast median is 0.035
mm^-1.

Correlations compare against another FBP on an arbitrary scale, so SART scores
lower even though its attenuation is physical. The held-out residual rejects the
reversed views clearly but separates the source position by at most 0.05
percentage points; the correlation with VICTRE's reconstruction is what pins the
source above the chest wall. This is one phantom with simulated physics and a
scalar flat field per view.

`dbt/calibrate_victre.py` tests whether the trainable geometry can recover the
arc from the projections alone. Each case starts from a geometry, reconstructs
with SIRT, then alternates one Adam step on the chosen parameters, against the
masked projection residual with the volume fixed, with two warm-started SIRT
iterations. VICTRE's FBP is only compared before and after, never optimised.

```bash
python benchmark/dbt/calibrate_victre.py \
    --input benchmark/dbt/results/victre/input_bin4.npz \
    --output benchmark/dbt/results/calibration/bin16
```

On 2026-10-19, phantom 694607065 at 1.36 mm detector bins, 50 geometry steps
with cosine-decayed learning rates, on an RTX 2080 Ti (about 8 minutes per case on
the PyTorch path; 0.7 s on the CUDA kernels, with the same results):

| Start | Learns | End | FBP correlation, start to end |
| --- | --- | --- | ---: |
| published | angles, radius, `source_y` | radius 596.7 mm, `source_y` -60.7 mm, angles up to 0.39 degrees off | 0.855 to 0.753 |
| angles +3 degrees | angles | mean error 2.79 degrees | 0.280 to 0.318 |
| `source_y` 0 mm | `source_y` | -2.0 mm, published -65.3 mm | 0.589 to 0.596 |
| radius 630 mm | radius | 610.5 mm, published 600 mm | 0.340 to 0.650 |

Only the radius moves substantially toward the published value. With 25 views
over 50 degrees, a re-fitted volume absorbs angle and source-position errors,
and from the correct start the fit drifts away and the image gets worse. The
projection residual falls in every case, mostly from the extra SIRT iterations,
so it cannot select the geometry. At the full 0.34 mm binning, which the
kernels' geometry gradient makes practical (7.6 s), the published start drifts the
same way: radius 597.5 mm, `source_y` -61.8 mm, FBP correlation 0.896 to 0.663.
Calibration from projections alone is therefore unreliable for this geometry; the trainable geometry is better suited
to a known calibration object, as in the synthetic test in `tests/test_dbt.py`.

Please cite Badano et al., "Evaluation of Digital Breast Tomosynthesis as
Replacement of Full-Field Digital Mammography Using an In Silico Imaging Trial",
JAMA Network Open 2018, and the TCIA VICTRE collection,
DOI [10.7937/TCIA.2019.ho23nxaw](https://doi.org/10.7937/TCIA.2019.ho23nxaw).

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
