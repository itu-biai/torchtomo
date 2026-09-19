"""LEAP-backed drop-ins for torchtomo's ParallelBeam and FanBeam.

Subclassing keeps every geometry buffer, the circle mask, and the angle list
exactly as torchtomo defines them, and replaces only the four operators, so a run
that swaps this in differs from an ordinary run in the projector kernels alone.

LEAP turns the gantry the other way round; negating the arc is the only convention
that has to change. With the image and detector both on [-1, 1] the two forward
projectors then agree to about 0.1% in relative L2, with no fitted scale factor.

LEAP is an optional benchmark dependency. Nothing in torchtomo or in
examples/ellipses imports this module unless it is asked for by name.
"""

import numpy as np
import torch
from leapctype import tomographicModels

from torchtomo import FanBeam, ParallelBeam


class _Project(torch.autograd.Function):
    """A x, with LEAP's backprojection as the gradient: the two are a matched pair."""

    @staticmethod
    def forward(ctx, image, projector):
        ctx.projector = projector
        return projector.project_raw(image)

    @staticmethod
    def backward(ctx, gradient):
        return ctx.projector.backproject_raw(gradient.contiguous()), None


class _Backproject(torch.autograd.Function):
    """A^T y, with LEAP's forward projection as the gradient."""

    @staticmethod
    def forward(ctx, sinogram, projector):
        ctx.projector = projector
        return projector.backproject_raw(sinogram)

    @staticmethod
    def backward(ctx, gradient):
        return ctx.projector.project_raw(gradient.contiguous()), None


class LeapParallelBeam(ParallelBeam):
    """ParallelBeam with LEAP's CUDA kernels behind forward, adjoint, and FBP."""

    # LEAP's own default is its Shepp-Logan ramp (order 2); 12 is Ram-Lak, which is
    # what torchtomo's "ramp" filter is, so the two FBPs differ in kernel and not in
    # the filter they apply.
    RAM_LAK = 12

    def __init__(
        self,
        img_size=256,
        n_angles=180,
        n_det=None,
        angle_range=(0, np.pi),
        circle=True,
        gpu=0,
        ramp_filter=RAM_LAK,
        angles=None,
        **kwargs,
    ):
        super().__init__(img_size, n_angles, n_det, angle_range, circle, angles=angles, **kwargs)
        if self.n_det != img_size:
            raise ValueError("the LEAP geometry here assumes a detector as wide as the image")
        self.gpu = gpu
        self.ramp_filter = ramp_filter
        self._models = {}

    def compute_device(self, tensor):
        """Where LEAP will actually run.

        LEAP's CPU parallel-beam kernel faults on a volume of more than one slice,
        so CPU tensors are worked on the card and handed back where they came from.
        """
        if tensor.is_cuda:
            return tensor.device
        if not torch.cuda.is_available():
            raise RuntimeError("LEAP needs CUDA here: its CPU parallel-beam kernel faults on multi-slice volumes")
        return torch.device("cuda", self.gpu)

    def leap_model(self, slices, device):
        """One LEAP geometry per slice count: LEAP carries that in the geometry itself."""
        key = (slices, device.type, device.index or 0)
        if key not in self._models:
            size = self.img_size
            pixel = 2.0 / size
            model = tomographicModels()
            model.set_gpu(device.index or 0 if device.type == "cuda" else -1)
            model.print_warnings = False
            phis = np.ascontiguousarray(-np.degrees(self.angles.detach().cpu().numpy()), dtype=np.float32)
            model.set_parallelbeam(len(phis), slices, size, pixel, pixel, (slices - 1) / 2.0, (size - 1) / 2.0, phis)
            model.set_volume(size, size, slices, pixel, pixel)
            model.set_rampFilter(self.ramp_filter)
            self._models[key] = model
        return self._models[key]

    @torch.no_grad()
    def project_raw(self, image):
        """[B, 1, H, W] -> [B, 1, angles, detectors], LEAP's own layout in between."""
        batch = image.shape[0]
        device = self.compute_device(image)
        model = self.leap_model(batch, device)
        sinogram = torch.zeros(self.n_angles, batch, self.n_det, device=device)
        model.project(sinogram, image[:, 0].to(device).contiguous())
        return sinogram.permute(1, 0, 2).unsqueeze(1).contiguous().to(image.device)

    @torch.no_grad()
    def backproject_raw(self, sinogram):
        """[B, 1, angles, detectors] -> [B, 1, H, W]."""
        batch = sinogram.shape[0]
        device = self.compute_device(sinogram)
        model = self.leap_model(batch, device)
        volume = torch.zeros(batch, self.img_size, self.img_size, device=device)
        model.backproject(sinogram[:, 0].to(device).permute(1, 0, 2).contiguous(), volume)
        return volume.unsqueeze(1).to(sinogram.device)

    def forward(self, x):
        if self.circle:
            x = x * self.circle_mask.view(1, 1, self.img_size, self.img_size)
        return _Project.apply(x.contiguous(), self)

    def backward(self, sinogram):
        # forward() projects the masked image, so the matching adjoint masks too.
        image = _Backproject.apply(sinogram.contiguous(), self)
        return image * self.circle_mask.view(1, 1, self.img_size, self.img_size) if self.circle else image

    def adjoint(self, sinogram):
        return self.backward(sinogram)

    @torch.no_grad()
    def backproject(self, sinogram):
        batch = sinogram.shape[0]
        device = self.compute_device(sinogram)
        model = self.leap_model(batch, device)
        volume = torch.zeros(batch, self.img_size, self.img_size, device=device)
        model.weightedBackproject(sinogram[:, 0].to(device).permute(1, 0, 2).contiguous().clone(), volume)
        image = volume.unsqueeze(1).to(sinogram.device)
        return image * self.circle_mask.view(1, 1, self.img_size, self.img_size) if self.circle else image

    @torch.no_grad()
    def fbp(self, sinogram, filter_name="ramp"):
        """LEAP's filtered backprojection. Its own ramp filter is the only one offered.

        Nothing in the benchmark differentiates through a reconstruction, so this
        stays outside the graph rather than pretending to an adjoint it does not have.
        """
        if filter_name != "ramp":
            raise NotImplementedError(f"LEAP's FBP here offers the ramp filter only, not {filter_name!r}")
        if sinogram.requires_grad:
            raise RuntimeError("LEAP's FBP is not differentiable here; use forward()/backward() for a matched pair")
        batch = sinogram.shape[0]
        device = self.compute_device(sinogram)
        model = self.leap_model(batch, device)
        volume = torch.zeros(batch, self.img_size, self.img_size, device=device)
        model.FBP(sinogram[:, 0].to(device).permute(1, 0, 2).contiguous().clone(), volume)
        image = volume.unsqueeze(1).to(sinogram.device)
        return image * self.circle_mask.view(1, 1, self.img_size, self.img_size) if self.circle else image


class LeapFanBeam(FanBeam):
    """FanBeam with LEAP's CUDA kernels behind forward, adjoint, and FBP.

    LEAP's fan-beam y axis is opposite torchtomo's, so the volume is flipped
    at the kernel boundary. The flip is orthogonal, so the matched pair holds.
    """

    RAM_LAK = 12

    def __init__(
        self,
        img_size=256,
        n_angles=360,
        n_det=None,
        src_dist=None,
        det_dist=None,
        det_width=None,
        det_spacing=None,
        angle_range=(0, 2 * np.pi),
        n_samples=None,
        circle=True,
        gpu=0,
        ramp_filter=RAM_LAK,
        angles=None,
    ):
        super().__init__(
            img_size,
            n_angles,
            n_det,
            src_dist,
            det_dist,
            det_width,
            det_spacing,
            angle_range,
            n_samples,
            circle,
            angles=angles,
        )
        self.gpu = gpu
        self.ramp_filter = ramp_filter
        self._models = {}

    def compute_device(self, tensor):
        if tensor.is_cuda:
            return tensor.device
        if not torch.cuda.is_available():
            raise RuntimeError("LEAP needs CUDA here: its CPU kernel faults on multi-slice volumes")
        return torch.device("cuda", self.gpu)

    def leap_model(self, slices, device):
        key = (slices, device.type, device.index or 0)
        if key not in self._models:
            size = self.img_size
            pixel = 2.0 / size
            model = tomographicModels()
            model.set_gpu(device.index or 0 if device.type == "cuda" else -1)
            model.print_warnings = False
            phis = np.ascontiguousarray(-np.degrees(self.angles.detach().cpu().numpy()), dtype=np.float32)
            sod = self.src_dist * pixel
            sdd = (self.src_dist + self.det_dist) * pixel
            det_pitch = (self.det_width / self.n_det) * pixel
            model.set_fanbeam(
                len(phis),
                slices,
                self.n_det,
                pixel,
                det_pitch,
                (slices - 1) / 2.0,
                (self.n_det - 1) / 2.0,
                phis,
                sod,
                sdd,
                0.0,
            )
            model.set_volume(size, size, slices, pixel, pixel)
            model.set_rampFilter(self.ramp_filter)
            self._models[key] = model
        return self._models[key]

    @torch.no_grad()
    def project_raw(self, image):
        batch = image.shape[0]
        device = self.compute_device(image)
        model = self.leap_model(batch, device)
        sinogram = torch.zeros(self.n_angles, batch, self.n_det, device=device)
        model.project(sinogram, image[:, 0].to(device).contiguous().flip(-2))
        return sinogram.permute(1, 0, 2).unsqueeze(1).contiguous().to(image.device)

    @torch.no_grad()
    def backproject_raw(self, sinogram):
        batch = sinogram.shape[0]
        device = self.compute_device(sinogram)
        model = self.leap_model(batch, device)
        volume = torch.zeros(batch, self.img_size, self.img_size, device=device)
        model.backproject(sinogram[:, 0].to(device).permute(1, 0, 2).contiguous(), volume)
        return volume.flip(-2).unsqueeze(1).to(sinogram.device)

    def forward(self, x):
        if self.circle:
            x = x * self.circle_mask.view(1, 1, self.img_size, self.img_size)
        return _Project.apply(x.contiguous(), self)

    def backward(self, sinogram):
        image = _Backproject.apply(sinogram.contiguous(), self)
        return image * self.circle_mask.view(1, 1, self.img_size, self.img_size) if self.circle else image

    def adjoint(self, sinogram):
        return self.backward(sinogram)

    @torch.no_grad()
    def backproject(self, sinogram):
        batch = sinogram.shape[0]
        device = self.compute_device(sinogram)
        model = self.leap_model(batch, device)
        volume = torch.zeros(batch, self.img_size, self.img_size, device=device)
        model.weightedBackproject(sinogram[:, 0].to(device).permute(1, 0, 2).contiguous().clone(), volume)
        image = volume.flip(-2).unsqueeze(1).to(sinogram.device)
        return image * self.circle_mask.view(1, 1, self.img_size, self.img_size) if self.circle else image

    @torch.no_grad()
    def fbp(self, sinogram, filter_name="ramp"):
        if filter_name != "ramp":
            raise NotImplementedError(f"LEAP's FBP here offers the ramp filter only, not {filter_name!r}")
        if sinogram.requires_grad:
            raise RuntimeError("LEAP's FBP is not differentiable here; use forward()/backward() for a matched pair")
        batch = sinogram.shape[0]
        device = self.compute_device(sinogram)
        model = self.leap_model(batch, device)
        volume = torch.zeros(batch, self.img_size, self.img_size, device=device)
        model.FBP(sinogram[:, 0].to(device).permute(1, 0, 2).contiguous().clone(), volume)
        image = volume.flip(-2).unsqueeze(1).to(sinogram.device)
        return image * self.circle_mask.view(1, 1, self.img_size, self.img_size) if self.circle else image
