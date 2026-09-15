"""Pack a subset of the BIAI I2I CT dataset into three compact archives.

The raw dataset stores one multi-array npz per slice, which is far larger than the
ground truth alone. This writes train.npz, val.npz, and test.npz holding only the
ground truth attenuation images, so a run can be reproduced from a small directory.

    python examples/ellipses/pack_ct_subset.py \
        --source /path/to/dataset_biailab_I2I_2025v1 \
        --output examples/ellipses/ct-subset \
        --train 200 --val 50 --test 50
"""

import argparse
from pathlib import Path

import numpy as np


def select(files, count):
    """Take evenly spaced files so the subset spans the split's patients.

    The raw names sort by patient, so even spacing covers the patient range
    without needing a random seed.
    """
    if count >= len(files):
        return files
    positions = np.linspace(0, len(files) - 1, count).round().astype(int)
    return [files[index] for index in sorted(dict.fromkeys(positions))]


def recover_window(full, clipped):
    """Recover the display window this dataset applied to one slice.

    The stored clipped image is clip((full - lo) / (hi - lo), 0, 1), so a fit over
    the pixels strictly inside the window returns lo and hi exactly.
    """
    inside = (clipped > 0.02) & (clipped < 0.98)
    if inside.sum() < 100:
        raise ValueError("too few pixels inside the window to recover it")
    slope, intercept = np.polyfit(full[inside], clipped[inside], 1)
    low = -intercept / slope
    high = low + 1.0 / slope
    error = np.abs(np.clip((full - low) / (high - low), 0, 1) - clipped).max()
    if error > 1e-3:
        raise ValueError(f"recovered window does not reproduce the clipped image (error {error:.5f})")
    return low, high


def pack(source, destination, split, count, array):
    files = sorted((source / split).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files under {source / split}")
    chosen = select(files, count)
    images, windows = [], []
    for path in chosen:
        with np.load(path, allow_pickle=False) as archive:
            if array not in archive:
                raise KeyError(f"{path.name} has no '{array}' array; found {list(archive)}")
            images.append(archive[array].astype(np.float16))
            if "gt_clipped" in archive:
                windows.append(recover_window(archive[array].ravel(), archive["gt_clipped"].ravel()))
    stacked = np.stack(images)
    names = np.array([path.name for path in chosen])
    target = destination / f"{split}.npz"
    extra = {"windows": np.array(windows, dtype=np.float32)} if len(windows) == len(chosen) else {}
    np.savez_compressed(target, images=stacked, names=names, **extra)
    if extra:
        low, high = extra["windows"].T
        print(
            f"      windows: {len(set(zip(low, high)))} distinct, low {low.min():.4f}..{low.max():.4f}, "
            f"high {high.min():.4f}..{high.max():.4f}"
        )
    patients = {name.split("_")[0] for name in names}
    print(
        f"{split:5s} {len(chosen):4d} slices from {len(patients):3d} patients "
        f"of {len(files)} available  ->  {target.name} "
        f"({target.stat().st_size / 1e6:.1f} MB, {stacked.dtype}, {stacked.shape[1:]})"
    )
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Directory holding train/, val/, and test/")
    parser.add_argument("--output", type=Path, required=True, help="Directory to write the packed splits into")
    parser.add_argument("--array", default="gt_full", help="Array to read from each raw npz")
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--val", type=int, default=50)
    parser.add_argument("--test", type=int, default=50)
    args = parser.parse_args()

    if min(args.train, args.val, args.test) < 1:
        parser.error("each split needs at least one slice")
    args.output.mkdir(parents=True, exist_ok=True)

    everything = {}
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        everything[split] = pack(args.source, args.output, split, count, args.array)

    for left in ("train", "val", "test"):
        for right in ("train", "val", "test"):
            if left < right:
                shared = {n.split("_")[0] for n in everything[left]} & {n.split("_")[0] for n in everything[right]}
                if shared:
                    raise RuntimeError(f"{left} and {right} share patients: {sorted(shared)[:5]}")
    print("splits are patient-disjoint")


if __name__ == "__main__":
    main()
