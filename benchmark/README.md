# Benchmarks

Accuracy and speed checks of torchtomo itself. Nothing here is needed to use
`torchtomo`, and the library never imports any of it. scikit-image, in the
`benchmark` extra, is the reference the accuracy checks measure against.

| File | What it does |
| --- | --- |
| `benchmark_speed.py` | forward, adjoint, and FBP time for every device and backend available |
| `benchmark_accuracy.py` | reconstruction quality on analytic phantoms, against scikit-image |
| `benchmark_adjoint.py` | inner-product test of `forward()` against `adjoint()` |
| `test_skimage_consistency.py` | agreement with `skimage.transform.radon` |
| `visual_comparison.py` | torchtomo and scikit-image reconstructions side by side at 512 px |
| `visualize.py` | parallel- and fan-beam reconstructions with their PSNR and SSIM |

```bash
pip install -e ".[benchmark]"
make benchmark          # pytest benchmark/
make benchmark-speed    # the backend speed table
PYTHONPATH=src python benchmark/benchmark_adjoint.py --pairs 500 --dtype float64
```

Comparisons against other projectors (LEAP, torch-radon), the reconstruction
methods trained on top of torchtomo (FBP+U-Net, Learned Primal-Dual, and the
rest), and their recorded results live in
[torchtomo-benchmark](https://github.com/itu-biai/torchtomo-benchmark).
