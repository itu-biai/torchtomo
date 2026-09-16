# Benchmarks

Comparisons against other projectors. Nothing here is needed to use `torchtomo`,
and neither the library nor `examples/ellipses` imports any of it by default.

| File | What it does |
| --- | --- |
| `benchmark_speed.py` | forward and back projection throughput, against skimage and torch-radon |
| `benchmark_accuracy.py` | reconstruction quality on analytic phantoms |
| `benchmark_adjoint.py` | inner-product test of `forward()` against `adjoint()` |
| `test_skimage_consistency.py` | agreement with `skimage.transform.radon` |
| `test_torchradon_consistency.py` | agreement with torch-radon |
| `test_leap_consistency.py` | agreement with LEAP |
| `leap_projector.py` | `LeapParallelBeam`, a drop-in `ParallelBeam` backed by LEAP's kernels |
| `leap_compare.py` | LEAP's own FBP, SIRT, and SART scored on a finished run's data |
| `compare_libraries.py` | torchtomo against LEAP and torch-radon: PSNR, SSIM, speed, GPU memory, and a figure |

## LEAP

[LEAP](https://github.com/LLNL/LEAP) (LivermorE AI Projector, LLNL) is an
independent CUDA implementation, so it checks the parallel-beam geometry including
its absolute scale: with the image and detector both on `[-1, 1]`, the two forward
projectors agree without any fitted factor. LEAP turns the gantry the other way
round, which is the one convention `leap_projector.py` negates.

It is not on PyPI, so it is built from source. On CUDA 11.5 with GCC 11 the build
needs two changes, both in `src/CMakeLists.txt`:

```bash
git clone --depth 1 https://github.com/LLNL/LEAP.git && cd LEAP
sed -i 's/find_package(CUDA 11.7 REQUIRED)/find_package(CUDA 11.5 REQUIRED)/' src/CMakeLists.txt
# all-major builds every architecture; pin your own to keep the build short
sed -i 's/CUDA_ARCHITECTURES all-major/CUDA_ARCHITECTURES 75/' src/CMakeLists.txt
pip install cmake
mkdir -p build && cd build
# C++17 is what breaks: nvcc 11.5 cannot parse GCC 11's <functional> under it
cmake .. -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc -DCMAKE_CUDA_STANDARD=14
cmake --build . -j 8
cd .. && pip install . --no-build-isolation
```

`setup.py` recompiles from scratch, so stub `etc/build.sh` with `exit 0` before the
last step to keep the library that was just built.

Two LEAP behaviours matter when comparing:

- its default ramp filter is Shepp-Logan (order 2), not Ram-Lak. `LeapParallelBeam`
  pins order 12, which is what torchtomo's `"ramp"` is.
- its CPU parallel-beam kernel faults on a volume of more than one slice, so
  `LeapParallelBeam` does CPU-resident work on the card and hands the result back.

### Running the comparisons

```bash
# agreement of the operators
PYTHONPATH=src pytest benchmark/test_leap_consistency.py

# LEAP's own untrained methods, on the data a finished run wrote
PYTHONPATH=src:examples/ellipses python benchmark/leap_compare.py \
    --results examples/ellipses/results-ctw

# the whole ten-method benchmark, on LEAP's kernels instead of torchtomo's
PYTHONPATH=src:examples/ellipses:benchmark python examples/ellipses/train.py \
    --projector leap --output examples/ellipses/results-ctw-leap ...
```

## torch-radon

torch-radon 1.0 predates two removals it depends on, `np.int` and `torch.rfft`, so
its `filter_sinogram` raises on any modern PyTorch. `compare_libraries.py` restores
both in its own process rather than editing the installed package, and it still
builds the filter with torch-radon's own `construct_fourier_filter`, so only the
transform calls differ from what the library shipped.

Its sinograms are torchtomo's divided by the pixel width: the measured scale is
exactly `2 / size` at every size tested, because torch-radon sums pixel values
where torchtomo integrates over a pixel of physical width.

## Three-way comparison

```bash
PYTHONPATH=src:benchmark python benchmark/compare_libraries.py --output benchmark-results
```

Each library projects the same phantom and reconstructs its own sinogram, so the
scale each one works in cancels and the quality figures need no fitted correction.
Speed and memory are only meaningful on an idle card; `--sections quality figure`
runs the parts that are not.

GPU memory is reported two ways. LEAP allocates outside PyTorch's caching
allocator, so `torch.cuda.max_memory_allocated` cannot see it; the driver-level
figure from `torch.cuda.mem_get_info` is the one that covers all three.
