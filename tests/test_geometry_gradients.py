"""Gradients with respect to the geometry: the pose table and what it moves."""

import pytest
import torch

from torchtomo import FanBeam, ParallelBeam


def _parallel(**kwargs):
    defaults = dict(img_size=16, n_angles=5, angle_range=(0.13, 2.71), circle=False)
    return ParallelBeam(**{**defaults, **kwargs}).double()


def _fan(**kwargs):
    # A detector narrow enough that every ray crosses the unit circle well inside
    # it: a ray that grazes the edge has a square root with an unbounded slope,
    # which no finite difference and no gradcheck can follow.
    defaults = dict(img_size=16, n_angles=5, n_det=9, det_width=12.0, angle_range=(0.13, 2.71), circle=False)
    return FanBeam(**{**defaults, **kwargs}).double()


GEOMETRIES = {"parallel": _parallel, "fan": _fan}


def _write_pose(projector, pose):
    """Set every column of the pose through the public writer."""
    names = {"angle": "angles"}
    columns: dict[str, torch.Tensor] = {}
    for index, name in enumerate(projector._POSE_COLUMNS):
        columns[names.get(name, name)] = pose[:, index]
    projector.set_pose(**columns)


def _finite_difference(projector, loss, base, eps=1e-6):
    """Central differences of `loss` in every pose entry, geometry written properly."""
    out = torch.zeros_like(base)
    for view in range(base.shape[0]):
        for column in range(base.shape[1]):
            for sign in (1, -1):
                pose = base.clone()
                pose[view, column] += sign * eps
                _write_pose(projector, pose)
                with torch.no_grad():
                    out[view, column] += sign * float(loss()) / (2 * eps)
    _write_pose(projector, base)
    return out


class TestPoseTable:
    @pytest.mark.parametrize("name", GEOMETRIES)
    def test_angles_are_column_zero(self, name):
        projector = GEOMETRIES[name]()
        assert projector.pose.shape == (5, len(projector._POSE_COLUMNS))
        torch.testing.assert_close(projector.angles, projector.pose[:, 0])
        assert torch.count_nonzero(projector.pose[:, 1:]) == 0

    @pytest.mark.parametrize("name", GEOMETRIES)
    def test_double_keeps_the_float64_angles(self, name):
        """A pose is carried through .double(); its angles come from the master list."""
        plain = GEOMETRIES[name]()
        learnable = GEOMETRIES[name](learnable_geometry=True)
        assert torch.equal(plain.angles, learnable.angles)

    def test_detector_shift_translates_the_sinogram(self):
        """A whole-pixel shift moves the sinogram by one bin, the operator's own definition."""
        projector = _parallel(img_size=32, n_angles=4)
        image = torch.rand(1, 1, 32, 32, dtype=torch.float64)
        reference = projector.forward(image)
        projector.set_pose(detector_shift=torch.ones(4, dtype=torch.float64))
        shifted = projector.forward(image)
        # Interior only: one bin leaves the detector at each end.
        torch.testing.assert_close(shifted[..., 1:-2], reference[..., 2:-1])

    def test_fan_detector_shift_moves_the_rays_one_bin(self):
        """A fan ray is the line from the source to a detector point, and nothing else.

        With one bin per pixel, shifting the detector by a pixel puts bin i on the
        line that bin i + 1 used, which is an exact identity rather than a resampling.
        """
        projector = _fan(img_size=16, n_angles=4, n_det=33, det_width=32.0)
        image = torch.rand(1, 1, 16, 16, dtype=torch.float64)
        reference = projector.forward(image)
        projector.set_pose(detector_shift=torch.ones(4, dtype=torch.float64))
        shifted = projector.forward(image)
        torch.testing.assert_close(shifted[..., :-1], reference[..., 1:])

    @pytest.mark.parametrize("name", GEOMETRIES)
    def test_learnable_geometry_is_a_parameter(self, name):
        projector = GEOMETRIES[name](learnable_geometry=True)
        assert isinstance(projector.pose, torch.nn.Parameter)
        assert any(p is projector.pose for p in projector.parameters())

    @pytest.mark.parametrize("name", GEOMETRIES)
    def test_an_optimiser_step_is_never_stale(self, name):
        """In-place writes under no_grad must reach the next call, cached or not."""
        projector = GEOMETRIES[name](learnable_geometry=True)
        image = torch.rand(1, 1, 16, 16, dtype=torch.float64)
        before = projector.forward(image).detach().clone()
        with torch.no_grad():
            projector.pose[:, 1] += 2.0
            under_no_grad = projector.forward(image).clone()
        assert not torch.allclose(before, under_no_grad)
        torch.testing.assert_close(under_no_grad, projector.forward(image).detach())

    def test_a_fixed_geometry_still_caches_its_grids(self):
        """The cost guard: nothing about the pose table reaches a projector that is fixed."""
        projector = _parallel(img_size=32, n_angles=4)
        image = torch.rand(1, 1, 32, 32, dtype=torch.float64)
        projector.forward(image)
        cached = len(projector._grid_cache)
        assert cached > 0
        projector.forward(image)
        assert len(projector._grid_cache) == cached

    def test_a_learnable_fan_never_builds_the_whole_grid(self):
        """The fan's four grids are 2.2 GB at 512 px; a moving pose rebuilds per chunk."""
        fixed, learnable = _fan(), _fan(learnable_geometry=True)
        image = torch.rand(1, 1, 16, 16, dtype=torch.float64)
        fixed.forward(image)
        learnable.forward(image)
        assert fixed._buffers["ray_grids"] is not None
        assert learnable._buffers["ray_grids"] is None

    def test_a_learnable_pose_never_reads_a_stale_matrix(self):
        """sparse_adjoint builds its CSR matrix once; a moving pose must not use it."""
        sparse = _parallel(learnable_geometry=True, sparse_adjoint=True)
        dense = _parallel(learnable_geometry=True)
        sinogram = torch.rand(1, 1, 5, 16, dtype=torch.float64)
        with torch.no_grad():
            sparse.adjoint(sinogram)
            for projector in (sparse, dense):
                projector.pose[:, 1] += 1.5
            torch.testing.assert_close(sparse.adjoint(sinogram), dense.adjoint(sinogram))

    def test_set_pose_rejects_a_column_this_geometry_lacks(self):
        with pytest.raises(ValueError, match="source_shift"):
            _parallel().set_pose(source_shift=1.0)

    def test_writing_the_pose_drops_the_fan_grids(self):
        projector = _fan()
        projector.forward(torch.rand(1, 1, 16, 16, dtype=torch.float64))
        assert projector._buffers["ray_grids"] is not None
        projector.set_pose(detector_shift=0.5)
        assert projector._buffers["ray_grids"] is None


class TestPoseGradients:
    @pytest.mark.parametrize("name", GEOMETRIES)
    def test_forward_gradient_matches_finite_differences(self, name):
        torch.manual_seed(0)
        projector = GEOMETRIES[name]()
        image = torch.rand(1, 1, 16, 16, dtype=torch.float64)
        weight = torch.rand(1, 1, 5, projector.n_det, dtype=torch.float64)
        base = projector.pose.detach().clone()

        projector.pose.requires_grad_(True)
        (analytic,) = torch.autograd.grad((projector.forward(image) * weight).sum(), projector.pose)
        numeric = _finite_difference(projector, lambda: (projector.forward(image) * weight).sum(), base)

        assert analytic.abs().max() > 0.1  # the test would pass on two zeros otherwise
        torch.testing.assert_close(analytic, numeric, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("name", GEOMETRIES)
    def test_adjoint_gradient_matches_finite_differences(self, name):
        """A^T y differentiates the geometry too, through the throwaway forward."""
        torch.manual_seed(0)
        projector = GEOMETRIES[name]()
        sinogram = torch.rand(1, 1, 5, projector.n_det, dtype=torch.float64)
        probe = torch.rand(1, 1, 16, 16, dtype=torch.float64)
        base = projector.pose.detach().clone()

        projector.pose.requires_grad_(True)
        (analytic,) = torch.autograd.grad((projector.adjoint(sinogram) * probe).sum(), projector.pose)
        numeric = _finite_difference(projector, lambda: (projector.adjoint(sinogram) * probe).sum(), base)

        assert analytic.abs().max() > 0.1
        torch.testing.assert_close(analytic, numeric, atol=1e-6, rtol=1e-5)

    def test_gradcheck_forward_parallel(self):
        projector = ParallelBeam(img_size=4, n_angles=3, angle_range=(0.2, 2.4), circle=False).double()
        image = torch.rand(1, 1, 4, 4, dtype=torch.float64)
        pose = projector.pose.detach().clone().requires_grad_(True)

        def project(pose_input):
            projector.pose = pose_input
            return projector.forward(image)

        assert torch.autograd.gradcheck(project, (pose,), eps=1e-6, atol=1e-8)

    def test_gradcheck_forward_fan(self):
        projector = _fan(img_size=4, n_angles=3, n_det=5, det_width=3.0, n_samples=4)
        image = torch.rand(1, 1, 4, 4, dtype=torch.float64)
        pose = projector.pose.detach().clone().requires_grad_(True)

        def project(pose_input):
            projector.pose = pose_input
            return projector.forward(image)

        assert torch.autograd.gradcheck(project, (pose,), eps=1e-6, atol=1e-8)

    @pytest.mark.parametrize("operator", ["forward", "adjoint"])
    @pytest.mark.parametrize("name", GEOMETRIES)
    def test_gradgradcheck(self, name, operator):
        """The PyTorch path differentiates the geometry a second time as well."""
        tiny = dict(img_size=4, n_angles=3, angle_range=(0.2, 2.4))
        if name == "fan":
            tiny.update(n_det=5, det_width=3.0, n_samples=4)
        projector = GEOMETRIES[name](**tiny)
        torch.manual_seed(0)
        image = torch.rand(1, 1, 4, 4, dtype=torch.float64)
        sinogram = torch.rand(1, 1, 3, projector.n_det, dtype=torch.float64)
        pose = projector.pose.detach().clone().requires_grad_(True)

        def operate(pose_input):
            projector.pose = pose_input
            return projector.forward(image) if operator == "forward" else projector.adjoint(sinogram)

        assert torch.autograd.gradgradcheck(operate, (pose,), eps=1e-6, atol=1e-6)

    @pytest.mark.parametrize("name", GEOMETRIES)
    def test_image_gradient_survives(self, name):
        """Differentiating the geometry must not cost the gradient we already had."""
        torch.manual_seed(0)
        projector = GEOMETRIES[name]()
        image = torch.rand(1, 1, 16, 16, dtype=torch.float64, requires_grad=True)
        projector.pose.requires_grad_(True)
        weight = torch.rand(1, 1, 5, projector.n_det, dtype=torch.float64)

        loss = (projector.forward(image) * weight).sum()
        image_grad, pose_grad = torch.autograd.grad(loss, (image, projector.pose))

        # dL/dx is A^T w, whatever the geometry is doing.
        torch.testing.assert_close(image_grad, projector.adjoint(weight).detach())
        assert pose_grad.abs().max() > 0


def _distance_difference(projector, loss, eps=1e-6):
    """Central differences of `loss` in src_dist, det_dist and det_width."""
    base = list(projector._distance_floats())
    out = torch.zeros(3, dtype=torch.float64)
    for index in range(3):
        for sign in (1, -1):
            values = list(base)
            values[index] += sign * eps
            projector.set_distances(*values)
            with torch.no_grad():
                out[index] += sign * float(loss()) / (2 * eps)
    projector.set_distances(*base)
    return out


class TestFanDistances:
    def test_distances_are_part_of_the_geometry(self):
        fixed, learnable = _fan(), _fan(learnable_geometry=True)
        assert "distances" in fixed.state_dict()
        assert not isinstance(fixed.distances, torch.nn.Parameter)
        assert any(p is learnable.distances for p in learnable.parameters())
        torch.testing.assert_close(
            fixed.distances, torch.tensor([32.0, 32.0, 12.0], dtype=torch.float64), rtol=0, atol=0
        )
        assert (fixed.src_dist, fixed.det_dist, fixed.det_width) == (32.0, 32.0, 12.0)

    def test_set_distances_is_a_fresh_projector(self):
        image = torch.rand(1, 1, 16, 16, dtype=torch.float64)
        moved = _fan()
        moved.forward(image)
        moved.set_distances(src_dist=41.5, det_width=13.25)
        fresh = _fan(src_dist=41.5, det_width=13.25)
        assert torch.equal(moved.forward(image), fresh.forward(image))
        assert torch.equal(moved.fbp(fresh.forward(image)), fresh.fbp(fresh.forward(image)))

    @pytest.mark.parametrize("operator", ["forward", "adjoint", "fbp"])
    def test_distance_gradient_matches_finite_differences(self, operator):
        torch.manual_seed(0)
        projector = _fan()
        image = torch.rand(1, 1, 16, 16, dtype=torch.float64)
        sinogram = torch.rand(1, 1, 5, projector.n_det, dtype=torch.float64)
        weight = torch.rand_like(sinogram if operator == "forward" else image)

        def loss():
            if operator == "forward":
                out = projector.forward(image)
            elif operator == "adjoint":
                out = projector.adjoint(sinogram)
            else:
                out = projector.fbp(sinogram)
            return (out * weight).sum()

        projector.distances.requires_grad_(True)
        (analytic,) = torch.autograd.grad(loss(), projector.distances)
        projector.distances.requires_grad_(False)
        numeric = _distance_difference(projector, loss)
        assert analytic.abs().max() > 1e-3
        torch.testing.assert_close(analytic, numeric, atol=1e-6, rtol=1e-5)

    def test_gradcheck_distances(self):
        projector = _fan(img_size=4, n_angles=3, n_det=5, det_width=3.0, n_samples=4)
        image = torch.rand(1, 1, 4, 4, dtype=torch.float64)
        distances = projector.distances.detach().clone().requires_grad_(True)

        def project(values):
            projector.distances = values
            return projector.forward(image)

        assert torch.autograd.gradcheck(project, (distances,), eps=1e-6, atol=1e-8)

    def test_an_optimiser_step_on_the_distances_is_never_stale(self):
        projector = _fan(learnable_geometry=True)
        image = torch.rand(1, 1, 16, 16, dtype=torch.float64)
        before = projector.forward(image).detach().clone()
        with torch.no_grad():
            projector.distances[0] += 3.0
            moved = projector.forward(image).clone()
            fbp = projector.fbp(moved)
        assert not torch.allclose(before, moved)
        assert projector.src_dist == 35.0
        reference = _fan(src_dist=35.0)
        torch.testing.assert_close(moved, reference.forward(image))
        torch.testing.assert_close(fbp, reference.fbp(moved))


class TestConversionsAndCheckpoints:
    def test_double_keeps_distances_float32_cannot_hold(self):
        projector = FanBeam(img_size=16, n_angles=4, src_dist=33.3).double()
        assert projector.distances[0].item() == 33.3

    @pytest.mark.parametrize("name", GEOMETRIES)
    def test_double_keeps_a_geometry_that_moved(self, name):
        """.double() restores angles as built, and only those: a trained pose stays trained."""
        projector = GEOMETRIES[name](learnable_geometry=True).float()
        with torch.no_grad():
            projector.pose[:, 0] += 0.01
        moved = projector.angles.detach().clone()
        projector.double()
        torch.testing.assert_close(projector.angles, moved.double())

    @pytest.mark.parametrize("name", GEOMETRIES)
    def test_a_0_3_checkpoint_still_loads(self, name):
        """0.3 saved `angles` and no pose or distances; strict loading must still work."""
        saved = GEOMETRIES[name]().state_dict()
        saved["angles"] = saved.pop("pose")[:, 0].clone()
        saved.pop("distances", None)
        projector = GEOMETRIES[name]()
        projector.load_state_dict(saved)
        torch.testing.assert_close(projector.angles, saved["angles"])

    @pytest.mark.parametrize("name", GEOMETRIES)
    def test_a_loaded_shift_is_used(self, name):
        """Loading copies into the pose in place; the projector must still see the shift."""
        shifted = GEOMETRIES[name]()
        shifted.set_pose(detector_shift=1.25)
        projector = GEOMETRIES[name]()
        projector.load_state_dict(shifted.state_dict())
        image = torch.rand(1, 1, 16, 16, dtype=torch.float64)
        torch.testing.assert_close(projector.forward(image), shifted.forward(image))
