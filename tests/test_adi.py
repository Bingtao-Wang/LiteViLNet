import unittest

import numpy as np

from litevilnet.data.adi import (
    altitude_gradient_7x7,
    interpolate_height_21x21,
    project_sparse_lidar_height,
)


class ADITest(unittest.TestCase):
    def test_projection_keeps_nearest_point_for_duplicate_pixel(self):
        projection = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        )
        points = np.array(
            [
                [20.0, 4.0, 4.0, 0.2],
                [10.0, 2.0, 2.0, 0.8],
            ],
            dtype=np.float32,
        )
        height, valid = project_sparse_lidar_height(points, projection, 4, 8)
        self.assertEqual(int(valid.sum()), 1)
        self.assertAlmostEqual(float(height[0, 4]), 2.0)

    def test_vectorized_interpolation_matches_direct_neighborhood(self):
        sparse = np.zeros((9, 11), dtype=np.float32)
        valid = np.zeros_like(sparse)
        sparse[1, 2], valid[1, 2] = 1.5, 1
        sparse[7, 9], valid[7, 9] = -0.5, 1
        actual_height, actual_valid = interpolate_height_21x21(sparse, valid)
        expected = np.zeros_like(sparse)
        for y in range(sparse.shape[0]):
            for x in range(sparse.shape[1]):
                if valid[y, x]:
                    expected[y, x] = sparse[y, x]
                    continue
                numerator = denominator = 0.0
                for yy in range(max(0, y - 10), min(sparse.shape[0], y + 11)):
                    for xx in range(max(0, x - 10), min(sparse.shape[1], x + 11)):
                        if not valid[yy, xx]:
                            continue
                        weight = 1.0 / max(np.hypot(yy - y, xx - x), 1.0)
                        numerator += sparse[yy, xx] * weight
                        denominator += weight
                expected[y, x] = numerator / denominator if denominator else 0.0
        np.testing.assert_allclose(actual_height, expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(actual_valid, np.ones_like(valid))

    def test_vectorized_gradient_matches_direct_7x7_loop(self):
        generator = np.random.default_rng(7)
        height = generator.normal(size=(8, 10)).astype(np.float32)
        valid = (generator.random((8, 10)) > 0.2).astype(np.float32)
        actual = altitude_gradient_7x7(height, valid)
        expected = np.zeros_like(height)
        for y in range(height.shape[0]):
            for x in range(height.shape[1]):
                if not valid[y, x]:
                    continue
                values = []
                for yy in range(max(0, y - 3), min(height.shape[0], y + 4)):
                    for xx in range(max(0, x - 3), min(height.shape[1], x + 4)):
                        if not valid[yy, xx]:
                            continue
                        difference = height[yy, xx] - height[y, x]
                        vertical = difference / (xx - x) if xx != x else 0.0
                        horizontal = difference / (yy - y) if yy != y else 0.0
                        values.append(np.hypot(vertical, horizontal))
                expected[y, x] = np.mean(values)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
