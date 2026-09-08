from __future__ import annotations

import unittest

import numpy as np
import torch

from frangi_filter.frangi_filter import FrangiFilter, symmetric_eigvalsh_3x3
from napari_sigma._geometry import exposed_face_measure
from napari_sigma.segmentation import _pairwise_potential, segmentation


class ScientificRegressions(unittest.TestCase):
    def test_exposed_faces(self):
        solid = np.ones((3, 3, 3), bool)
        self.assertEqual(exposed_face_measure(solid, (1, 1, 1)), 54)
        solid[1, 1, 1] = False
        self.assertEqual(exposed_face_measure(solid, (1, 1, 1)), 60)
        self.assertEqual(exposed_face_measure(solid, (3, .5, .5)), 65)
        self.assertEqual(exposed_face_measure(solid, (1, 1, 1), np.zeros_like(solid)), 0)

    def test_surface_selection_shape_validation(self):
        with self.assertRaises(ValueError):
            exposed_face_measure(np.ones((3, 3), bool), (1, 1), np.ones((1, 3), bool))

    def test_frangi_layout_and_constant_response(self):
        torch.manual_seed(42)
        for dim, shape in ((2, (1, 1, 24, 32)), (3, (1, 1, 7, 12, 16))):
            with self.subTest(dim=dim):
                model = FrangiFilter(1, 5, [1., 2.], dim, psf_ratio=1.)
                self.assertEqual(float(model(torch.ones(shape)).abs().max()), 0)
                image = -torch.rand(shape).transpose(-1, -2) * 255
                expected = model(image.contiguous())
                for _ in range(2):
                    torch.testing.assert_close(model(image), expected, rtol=0, atol=0)
                self.assertTrue(bool(torch.isfinite(expected).all()))
                self.assertLessEqual(float(expected.max()), 1.)

    def test_eigensolver_matches_reference(self):
        rng = np.random.default_rng(1)
        matrices = rng.normal(size=(1000, 3, 3)).astype(np.float32)
        matrices = (matrices + matrices.transpose(0, 2, 1)) / 2
        elements = [torch.from_numpy(matrices[:, i, j]) for i, j in
                    ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))]
        actual = symmetric_eigvalsh_3x3(elements).numpy().T
        np.testing.assert_allclose(actual, np.linalg.eigvalsh(matrices)[:, ::-1], atol=2e-5)

    def test_pairwise_buffers_match_independent_reference(self):
        devices = ["cpu"]
        if torch.backends.mps.is_available():
            devices.append("mps")
        if torch.cuda.is_available():
            devices.append("cuda")
        rng = np.random.default_rng(5)
        for shape in ((1, 13, 17), (1, 3, 13, 17)):
            labels = rng.integers(0, 2, size=shape, dtype=np.uint8)
            f0, f1 = [rng.uniform(-5, 0, shape).astype(np.float32) for _ in range(2)]
            smooth = np.zeros(shape, np.float32)
            for axis in range(1, len(shape)):
                weight = .35 if len(shape) == 4 and axis == 1 else .7
                padded = np.pad(labels.astype(np.float32), [(1, 1) if i == axis else (0, 0) for i in range(len(shape))])
                lo, hi = [slice(None)] * len(shape), [slice(None)] * len(shape)
                lo[axis], hi[axis] = slice(None, -2), slice(2, None)
                smooth += weight * (2 * padded[tuple(lo)] - 1) + weight * (2 * padded[tuple(hi)] - 1)
            expected = np.concatenate((smooth + f0, -smooth + f1), axis=0)
            for device in devices:
                with self.subTest(shape=shape, device=device):
                    output = torch.empty(expected.shape, device=device)
                    for _ in range(4):
                        actual = _pairwise_potential(torch.tensor(labels, device=device), .7, 1.,
                            torch.tensor(f0, device=device), torch.tensor(f1, device=device),
                            torch.device(device), .1, .2, output=output)
                        np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=2e-6)

    def test_segmentation_uint16_matches_float32(self):
        raw = np.arange(256, dtype=np.uint16).reshape(16, 16)
        frangi = raw.astype(np.float32) / 255
        kwargs = dict(pixel_size=(.1, .1), beta1=.5, beta2=1., n_fore=1, n_back=1,
                      max_iter=2, init_method="otsu")
        a, _ = segmentation(raw, frangi, **kwargs)
        b, _ = segmentation(raw.astype(np.float32), frangi, **kwargs)
        np.testing.assert_array_equal(a, b)

    def test_segmentation_rejects_invalid_inputs(self):
        for raw, pattern in ((np.ones((8, 8)), "constant"), (np.full((8, 8), np.nan), "finite"),
                             (np.empty((0, 8)), "nonempty")):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                segmentation(raw, np.zeros_like(raw), (1, 1), .5, 1.)
