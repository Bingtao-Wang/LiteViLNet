import unittest

import numpy as np

from litevilnet.data.robot_road_dataset import encode_depth3


class RobotRoadDepth3Test(unittest.TestCase):
    def test_depth3_uses_strict_validity_and_clipped_normalized_depth(self):
        depth = np.asarray([[0, 6000, 11999, 12000, 13000]], dtype=np.uint16)
        actual = encode_depth3(depth, max_depth_mm=12000.0)

        normalized = np.asarray(
            [[0.0, 0.5, 11999.0 / 12000.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        valid = np.asarray([[0.0, 1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        inverse = valid * (1.0 - normalized)
        expected = np.stack([normalized, valid, inverse], axis=-1)
        expected = (expected - 0.5) / 0.5

        np.testing.assert_allclose(actual, expected, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
