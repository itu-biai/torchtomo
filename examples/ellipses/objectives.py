"""Training objectives, so supervised and self-supervised methods share one loop.

FBP+U-Net, LPD, and iRadonMAP minimise the error against the clean image.
Noise2Inverse and Proj2Proj never see a clean image: they build their target out
of the measurements themselves, and their checkpoints are selected on their own
validation loss rather than on anything computed from the ground truth.
"""

import torch
from classical import subset_projectors
from torch.nn import functional as F


class Supervised:
    """Minimise the error against the clean image, the usual paired setting."""

    uses_truth = True
    criterion = "image MSE against the clean image"

    def __init__(self, inputs, truth):
        self.inputs = inputs
        self.truth = truth

    def loss(self, model, ids, device, step):
        return F.mse_loss(model(self.inputs[ids].to(device)), self.truth[ids].to(device))

    @torch.no_grad()
    def validation(self, model, ids, device, batch_size):
        model.eval()
        total = 0.0
        for batch in ids.split(batch_size):
            total += F.mse_loss(model(self.inputs[batch].to(device)), self.truth[batch].to(device)).item() * len(batch)
        return total / len(ids)

    @torch.no_grad()
    def reconstruct(self, model, ids, device, batch_size):
        model.eval()
        return torch.cat([model(self.inputs[batch].to(device)).cpu() for batch in ids.split(batch_size)])


class Noise2Inverse:
    """Hendriksen et al. 2020: learn between reconstructions of disjoint view subsets.

    Each subset is back-projected on its own, giving reconstructions that share a
    signal and carry independent noise. The network maps the mean of every subset
    but one onto the one left out, the paper's X:1 strategy, and at reconstruction
    time its outputs over all choices of the held-out subset are averaged.
    """

    uses_truth = False
    criterion = "held-out subset MSE"

    def __init__(self, projector, sinograms, splits):
        if projector.n_angles % splits:
            raise ValueError(f"{splits} subsets do not divide {projector.n_angles} angles evenly")
        self.sinograms = sinograms
        self.subsets = subset_projectors(projector, splits)
        self.splits = splits

    def reconstructions(self, sinogram):
        return [subset.fbp(sinogram[:, :, indices]) for indices, subset in self.subsets]

    def pair(self, sinogram, held_out):
        parts = self.reconstructions(sinogram)
        target = parts.pop(held_out)
        return torch.stack(parts).mean(0), target

    def loss(self, model, ids, device, step):
        inputs, target = self.pair(self.sinograms[ids].to(device), step % self.splits)
        return F.mse_loss(model(inputs), target)

    @torch.no_grad()
    def validation(self, model, ids, device, batch_size):
        model.eval()
        total = 0.0
        for batch in ids.split(batch_size):
            sinogram = self.sinograms[batch].to(device)
            for held_out in range(self.splits):
                inputs, target = self.pair(sinogram, held_out)
                total += F.mse_loss(model(inputs), target).item() * len(batch)
        return total / (len(ids) * self.splits)

    @torch.no_grad()
    def reconstruct(self, model, ids, device, batch_size):
        model.eval()
        outputs = []
        for batch in ids.split(batch_size):
            sinogram = self.sinograms[batch].to(device)
            parts = self.reconstructions(sinogram)
            averaged = torch.stack(
                [model(torch.stack(parts[:held] + parts[held + 1 :]).mean(0)) for held in range(self.splits)]
            )
            outputs.append(averaged.mean(0).cpu())
        return torch.cat(outputs)


def neighbour_average(sinogram):
    """Mean of the four edge neighbours, the value a masked entry is replaced by."""
    padded = F.pad(sinogram, (1, 1, 1, 1), mode="replicate")
    return (padded[..., :-2, 1:-1] + padded[..., 2:, 1:-1] + padded[..., 1:-1, :-2] + padded[..., 1:-1, 2:]) / 4


def perturb_projections(sinogram, index, grid):
    """Replace one position of every grid cell, the J-invariant mask of Noise2Self."""
    row, column = divmod(index, grid)
    mask = torch.zeros_like(sinogram, dtype=torch.bool)
    mask[..., row::grid, column::grid] = True
    return torch.where(mask, neighbour_average(sinogram), sinogram), mask


class Proj2Proj:
    """Unal et al. 2024: J-invariant self-supervision in the projection domain.

    One position in every grid cell of the sinogram is replaced by its neighbours'
    mean, that perturbed sinogram is reconstructed and denoised, and the result is
    forward projected. The loss compares it with the true measurements only where
    they were perturbed, which is what stops the network collapsing to the identity.
    """

    uses_truth = False
    criterion = "masked projection MSE"

    def __init__(self, projector, sinograms, grid=4):
        self.projector = projector
        self.sinograms = sinograms
        self.grid = grid
        # Scoring every mask position each epoch would cost more than the epoch
        # itself, so validation uses a fixed, evenly spread quarter of them.
        self.validation_masks = list(range(0, grid * grid, max(1, grid * grid // 4)))

    def masked_loss(self, model, sinogram, index):
        perturbed, mask = perturb_projections(sinogram, index, self.grid)
        image = model(self.projector.fbp(perturbed))
        residual = (self.projector.forward(image) - sinogram) * mask
        return residual.square().sum() / mask.sum()

    def loss(self, model, ids, device, step):
        return self.masked_loss(model, self.sinograms[ids].to(device), step % (self.grid * self.grid))

    @torch.no_grad()
    def validation(self, model, ids, device, batch_size):
        model.eval()
        total = 0.0
        for batch in ids.split(batch_size):
            sinogram = self.sinograms[batch].to(device)
            for index in self.validation_masks:
                total += self.masked_loss(model, sinogram, index).item() * len(batch)
        return total / (len(ids) * len(self.validation_masks))

    @torch.no_grad()
    def reconstruct(self, model, ids, device, batch_size):
        model.eval()
        outputs = []
        for batch in ids.split(batch_size):
            sinogram = self.sinograms[batch].to(device)
            outputs.append(model(self.projector.fbp(sinogram)).cpu())
        return torch.cat(outputs)
