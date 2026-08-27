import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satp.model import SATPParameters, exhaustion_weight, predict_raw, predict_satp


class ModelTests(unittest.TestCase):
    def test_raw_rating_curve(self):
        result = predict_raw([1.0, 4.0], a1=2.0, b1=0.5)
        np.testing.assert_allclose(result, [2.0, 4.0])

    def test_exhaustion_weight_is_finite(self):
        values = exhaustion_weight([0.0, 0.5, 1.0])
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values < 0).all())

    def test_satp_returns_nonnegative_concentrations(self):
        params = SATPParameters(8.0, -3.2, 1.7, 1.7, -3.0, -1.0, 6.6, -0.9)
        result = predict_satp([100.0, 200.0], [0.2, 0.8], [0.0, 0.5], [0.2, 0.8], params)
        self.assertEqual(result.shape, (2,))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result >= 0.005).all())

    def test_nonpositive_discharge_is_rejected(self):
        params = SATPParameters(1, 1, 1, 1, 1, 1, 1, 1)
        with self.assertRaises(ValueError):
            predict_satp([0.0], [0.0], [0.0], [0.0], params)


if __name__ == "__main__":
    unittest.main()

