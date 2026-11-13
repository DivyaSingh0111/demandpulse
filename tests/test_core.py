import math
import unittest

from app.anomaly import ResidualAnomalyDetector
from app.engine import DemandEngine
from app.forecaster import HoltWinters, mape


class ForecasterTest(unittest.TestCase):
    def test_learns_pure_seasonal_series(self):
        y = [100 + 20 * math.sin(2 * math.pi * t / 24) for t in range(24 * 10)]
        model = HoltWinters.auto_fit(y)
        truth = [100 + 20 * math.sin(2 * math.pi * t / 24) for t in range(240, 264)]
        self.assertLess(mape(truth, model.forecast(24)), 2.0)

    def test_requires_two_seasons(self):
        with self.assertRaises(ValueError):
            HoltWinters().fit([1.0] * 30)


class AnomalyTest(unittest.TestCase):
    def test_flags_spike_only(self):
        det = ResidualAnomalyDetector(min_points=10)
        for i in range(30):
            _, flag = det.score(100 + (i % 3 - 1), 100)
            self.assertFalse(flag)
        _, flag = det.score(200, 100)
        self.assertTrue(flag)


class EngineTest(unittest.TestCase):
    def test_tick_produces_snapshot(self):
        eng = DemandEngine(seed=1)
        eng.warmup()
        snap = eng.tick()
        self.assertEqual(len(snap["forecast_labels"]), 24)
        for c in snap["categories"].values():
            self.assertEqual(len(c["forecast"]), 24)
        self.assertGreater(snap["metrics"]["events_total"], 0)


if __name__ == "__main__":
    unittest.main()
