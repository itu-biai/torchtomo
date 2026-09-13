"""Small residual U-Net and a zero-initialized Learned Primal-Dual network."""

import torch
from torch import nn
from torch.nn import functional as F


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.PReLU(out_channels),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.PReLU(out_channels),
    )


class FBPUNet(nn.Module):
    """Two-level U-Net learning a residual correction to the unclipped FBP image."""

    def __init__(self, support, width=16):
        super().__init__()
        self.encoder1 = conv_block(1, width)
        self.encoder2 = conv_block(width, 2 * width)
        self.bottleneck = conv_block(2 * width, 4 * width)
        self.decoder2 = conv_block(6 * width, 2 * width)
        self.decoder1 = conv_block(3 * width, width)
        self.output = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("support", support[None, None].clone())

    def forward(self, fbp):
        e1 = self.encoder1(fbp)
        e2 = self.encoder2(F.avg_pool2d(e1, 2))
        middle = self.bottleneck(F.avg_pool2d(e2, 2))
        d2 = self.decoder2(torch.cat([F.interpolate(middle, size=e2.shape[-2:], mode="nearest"), e2], dim=1))
        d1 = self.decoder1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="nearest"), e1], dim=1))
        return (fbp + self.output(d1)) * self.support


class UpdateBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, width):
        super().__init__(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.PReLU(width),
            nn.Conv2d(width, width, 3, padding=1),
            nn.PReLU(width),
            nn.Conv2d(width, out_channels, 3, padding=1),
        )
        for layer in self:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        # Keep the initial unrolled updates small without cutting the dual gradients.
        nn.init.xavier_uniform_(self[-1].weight, gain=0.1)


class LearnedPrimalDual(nn.Module):
    """LPD with independent CNNs per iteration, primal/dual memory, and exact A^T.

    The operator and data are both divided by ||A||, and the adjoint uses the
    same scaling. States start at zero; this model does not receive an FBP.
    """

    def __init__(self, projector, operator_norm, iterations=5, memory=5, width=24):
        super().__init__()
        self.projector = projector
        self.memory = memory
        self.register_buffer("operator_norm", torch.tensor(float(operator_norm)))
        self.dual_updates = nn.ModuleList([UpdateBlock(memory + 2, memory, width) for _ in range(iterations)])
        self.primal_updates = nn.ModuleList([UpdateBlock(memory + 1, memory, width) for _ in range(iterations)])

    def forward(self, sinogram):
        batch = sinogram.shape[0]
        size = self.projector.img_size
        primal = sinogram.new_zeros(batch, self.memory, size, size)
        dual = sinogram.new_zeros(batch, self.memory, self.projector.n_angles, self.projector.n_det)
        data = sinogram / self.operator_norm
        for dual_update, primal_update in zip(self.dual_updates, self.primal_updates):
            projected = self.projector.forward(primal[:, 1:2]) / self.operator_norm
            dual = dual + dual_update(torch.cat([dual, projected, data], dim=1))
            backprojected = self.projector.backward(dual[:, :1]) / self.operator_norm
            primal = primal + primal_update(torch.cat([primal, backprojected], dim=1))
        return primal[:, :1] * self.projector.circle_mask[None, None]


@torch.no_grad()
def estimate_operator_norm(projector, iterations=25):
    """Power iteration on A^T A; depends on geometry, not on any dataset split."""
    x = torch.ones(1, 1, projector.img_size, projector.img_size, device=projector.angles.device)
    x /= x.norm()
    for _ in range(iterations):
        x = projector.backward(projector(x))
        x /= x.norm().clamp_min(1e-12)
    return projector(x).norm().item()
