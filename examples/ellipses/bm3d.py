"""BM3D in pure PyTorch, so the comparison needs no extra dependency.

Follows Dabov et al., Image denoising by sparse 3D transform-domain collaborative
filtering (IEEE TIP 2007): a hard thresholding stage produces a basic estimate,
then a Wiener stage filters the noisy image using that estimate's spectrum. Both
stages group similar blocks, transform the group with a 2D DCT and a 1D Haar
transform, shrink, and aggregate the results back with Kaiser windowing.

Reference blocks sit on a stride grid, but candidates are searched at every
pixel offset inside the search window, as in the original. Two simplifications
remain: every group holds exactly `group` blocks rather than a variable number
under a distance threshold, which keeps the group axis a power of two for the
Haar transform, and both stages use a DCT over each block where the reference
profile uses a bior1.5 wavelet in the hard thresholding stage.

Measured against the reference `bm3d` package on white Gaussian noise, with the
noise level tuned for each, this reaches 32.2 dB where the reference reaches
33.8 dB. Treat results from it as a lower bound on a tuned reference profile.
"""

import torch


def dct_matrix(size, device, dtype):
    """Orthonormal DCT-II, applied as a matrix because the blocks are tiny."""
    positions = torch.arange(size, device=device, dtype=dtype)
    frequencies = positions.view(-1, 1)
    matrix = torch.cos(torch.pi * (2 * positions.view(1, -1) + 1) * frequencies / (2 * size))
    matrix *= (2.0 / size) ** 0.5
    matrix[0] /= 2**0.5
    return matrix


def haar_matrix(size, device, dtype):
    """Orthonormal Haar transform for a power-of-two group axis."""
    if size & (size - 1):
        raise ValueError(f"group size {size} is not a power of two")
    matrix = torch.ones(1, 1, device=device, dtype=dtype)
    while matrix.shape[0] < size:
        current = matrix.shape[0]
        top = torch.kron(matrix, torch.tensor([1.0, 1.0], device=device, dtype=dtype))
        bottom = torch.kron(
            torch.eye(current, device=device, dtype=dtype), torch.tensor([1.0, -1.0], device=device, dtype=dtype)
        )
        matrix = torch.cat([top, bottom], dim=0) / 2**0.5
    return matrix


def kaiser_window(size, beta, device, dtype):
    window = torch.kaiser_window(size, periodic=False, beta=beta, device=device, dtype=dtype)
    return window.view(-1, 1) * window.view(1, -1)


def block_origins(length, block, step):
    """Stride positions, always including the last one so borders are covered."""
    origins = list(range(0, length - block + 1, step))
    if origins[-1] != length - block:
        origins.append(length - block)
    return torch.tensor(origins)


def all_blocks(image, block):
    """Every block position in the image, as [position, position, block, block]."""
    return image.unfold(0, block, 1).unfold(1, block, 1)


def match_offsets(windows, rows, columns, radius, group):
    """For every reference block, the pixel offsets of its `group` nearest neighbours.

    Candidates are taken at every offset in the search window, not only at the
    stride positions, which is what the reference implementation does and is
    worth several decibels. One displacement is scored at a time over all
    reference blocks at once.
    """
    reference = windows[rows][:, columns]
    last_row, last_column = windows.shape[0] - 1, windows.shape[1] - 1
    displacements, distances = [], []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            candidate = windows[(rows + dy).clamp(0, last_row)][:, (columns + dx).clamp(0, last_column)]
            distances.append((candidate - reference).square().mean(dim=(-2, -1)))
            displacements.append((dy, dx))
    order = torch.stack(distances, dim=-1).topk(group, dim=-1, largest=False).indices
    return torch.tensor(displacements, device=windows.device)[order]


def source_positions(windows, rows, columns, offsets):
    """Clamped block positions of every matched neighbour."""
    source_rows = (rows.view(-1, 1, 1) + offsets[..., 0]).clamp(0, windows.shape[0] - 1)
    source_columns = (columns.view(1, -1, 1) + offsets[..., 1]).clamp(0, windows.shape[1] - 1)
    return source_rows, source_columns


def gather_groups(windows, rows, columns, offsets):
    """Stack each reference block's matched neighbours along a new group axis."""
    source_rows, source_columns = source_positions(windows, rows, columns, offsets)
    return windows[source_rows, source_columns]


def transform(group, dct, haar):
    """2D DCT over each block, then a 1D Haar transform along the group axis."""
    spectrum = dct @ group @ dct.t()
    return torch.einsum("gk,...khw->...ghw", haar, spectrum)


def inverse_transform(spectrum, dct, haar):
    group = torch.einsum("kg,...khw->...ghw", haar, spectrum)
    return dct.t() @ group @ dct


def aggregate(shape, groups, weights, top, left, block, window):
    """Accumulate overlapping estimates, weighted, and normalise by the weights."""
    height, width = shape
    numerator = groups.new_zeros(height * width)
    denominator = groups.new_zeros(height * width)
    inside = torch.arange(block, device=groups.device)
    pixel_rows = top.unsqueeze(-1).unsqueeze(-1) + inside.view(-1, 1)
    pixel_columns = left.unsqueeze(-1).unsqueeze(-1) + inside.view(1, -1)
    flat_index = (pixel_rows * width + pixel_columns).reshape(-1)

    scaled = weights.reshape(*weights.shape, 1, 1, 1)
    weighted = (groups * window * scaled).reshape(-1)
    spread = (window * scaled).expand_as(groups).reshape(-1)
    numerator.index_add_(0, flat_index, weighted)
    denominator.index_add_(0, flat_index, spread)
    return (numerator / denominator.clamp_min(1e-12)).view(height, width)


def collaborative_filter(noisy, guide, sigma, block, step, radius, group, threshold, beta, hard):
    """One BM3D stage: match on `guide`, shrink `noisy`, aggregate."""
    device, dtype = noisy.device, noisy.dtype
    height, width = noisy.shape
    rows = block_origins(height, block, step)
    columns = block_origins(width, block, step)
    dct = dct_matrix(block, device, dtype)
    haar = haar_matrix(group, device, dtype)
    window = kaiser_window(block, beta, device, dtype)

    rows, columns = rows.to(device), columns.to(device)
    guide_windows = all_blocks(guide, block)
    noisy_windows = all_blocks(noisy, block)
    offsets = match_offsets(guide_windows, rows, columns, radius, group)
    noisy_groups = gather_groups(noisy_windows, rows, columns, offsets)
    noisy_spectrum = transform(noisy_groups, dct, haar)

    if hard:
        kept = noisy_spectrum.abs() > threshold * sigma
        shrunk = noisy_spectrum * kept
        retained = kept.sum(dim=(-3, -2, -1)).clamp_min(1)
        weights = 1.0 / retained.to(dtype)
    else:
        guide_spectrum = transform(gather_groups(guide_windows, rows, columns, offsets), dct, haar)
        energy = guide_spectrum.square()
        gain = energy / (energy + sigma**2)
        shrunk = noisy_spectrum * gain
        weights = 1.0 / gain.square().sum(dim=(-3, -2, -1)).clamp_min(1e-8)

    estimates = inverse_transform(shrunk, dct, haar)
    top, left = source_positions(guide_windows, rows, columns, offsets)
    return aggregate((height, width), estimates, weights, top, left, block, window)


def denoise(images, sigma, block=8, step=3, radius=8, hard_group=16, wiener_group=32, threshold=2.7, beta=2.0):
    """Two-stage BM3D over a batch of [B, 1, H, W] images, one image at a time."""
    outputs = []
    for image in images:
        basic = collaborative_filter(
            image[0], image[0], sigma, block, step, radius, hard_group, threshold, beta, hard=True
        )
        final = collaborative_filter(
            image[0], basic, sigma, block, step, radius, wiener_group, threshold, beta, hard=False
        )
        outputs.append(final[None])
    return torch.stack(outputs)
