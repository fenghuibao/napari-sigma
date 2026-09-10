from types import SimpleNamespace
import unittest

import numpy as np

from napari_sigma._proximity import compute_proximity_result


class ProximityRegressions(unittest.TestCase):
    def test_large_instance_ids_preserve_rows_and_geometry(self):
        for dims, shape, spacing in (("YX", (10, 12), (1., 1.)),
                                     ("TYX", (2, 10, 12), (1., 1.)),
                                     ("ZYX", (3, 10, 12), (2., 1., 1.)),
                                     ("TZYX", (2, 3, 10, 12), (2., 1., 1.))):
            small = np.zeros(shape, np.uint8)
            small[..., 1:3, 1:3] = 1
            small[..., 6:8, 7:9] = 2
            def layer(data):
                return SimpleNamespace(data=data, metadata={"dims": dims}, name="test")
            raw = layer(np.ones_like(small))
            target = layer((small > 0).astype(np.uint8))
            reference = compute_proximity_result(raw, layer(small), raw, target,
                                                 voxel_size=spacing, surface_only=False,
                                                 distance_threshold=0)
            for dtype, ids in ((np.uint32, (2**31, 2**31 + 1)),
                               (np.uint64, (2**63 + 1, 2**64 - 1))):
                with self.subTest(dims=dims, dtype=dtype):
                    source = np.zeros(shape, dtype=dtype)
                    source[small == 1], source[small == 2] = ids
                    source.setflags(write=False)
                    actual = compute_proximity_result(raw, layer(source), raw, target,
                                                      voxel_size=spacing, surface_only=False,
                                                      distance_threshold=0)
                    self.assertEqual([row.source_label for row in actual.component_rows], list(ids))
                    self.assertEqual([row.component_id for row in actual.component_rows], list(ids))
                    self.assertEqual(actual.summary_rows, reference.summary_rows)
                    self.assertEqual([row.size for row in actual.component_rows],
                                     [row.size for row in reference.component_rows])
                    np.testing.assert_array_equal(actual.component_labels, reference.component_labels)
                    np.testing.assert_array_equal(actual.source_object_labels, source)
                    self.assertEqual(actual.source_object_labels.dtype, source.dtype)

    def test_empty_masks_and_binary_time_frames_keep_existing_behavior(self):
        for occupied in (False, True):
            mask = np.zeros((2, 8, 9), np.uint8)
            if occupied:
                mask[:, 2:4, 2:4] = 255
            layer = SimpleNamespace(data=mask, metadata={"dims": "TYX"}, name="binary")
            result = compute_proximity_result(layer, layer, layer, layer,
                                              voxel_size=(1, 1), surface_only=False)
            self.assertEqual(len(result.component_rows), 2 if occupied else 0)
            if occupied:
                self.assertEqual([row.source_label for row in result.component_rows], [1, 2])
                # Full-image summary frame semantics are unchanged by ID remapping.
                self.assertEqual([row.frame for row in result.component_rows], [-1, -1])
