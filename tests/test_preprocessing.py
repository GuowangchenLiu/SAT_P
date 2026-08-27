import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satp.preprocessing import PreprocessingConfig, prepare_daily_inputs


class PreprocessingTests(unittest.TestCase):
    def test_predictors_are_created(self):
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2022-01-01", periods=10),
                "temperature": [273.15 + value for value in range(10)],
                "precipitation": [1.0] * 10,
                "discharge": [1, 2, 3, 5, 4, 8, 7, 9, 10, 11],
                "TP": [None] * 10,
            }
        )
        result = prepare_daily_inputs(frame, PreprocessingConfig())
        self.assertIn("temperature_8d_norm", result)
        self.assertIn("discharge_increase_2d_norm", result)
        self.assertIn("exhaustion_index", result)
        self.assertAlmostEqual(result.loc[9, "exhaustion_index"], 1.0)
        self.assertTrue(result["temperature_8d_norm"].between(0, 1).all())
        self.assertTrue(result["discharge_increase_2d_norm"].between(0, 1).all())

    def test_missing_required_column_is_rejected(self):
        frame = pd.DataFrame({"date": ["2022-01-01"]})
        with self.assertRaises(ValueError):
            prepare_daily_inputs(frame)


if __name__ == "__main__":
    unittest.main()

