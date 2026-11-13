"""Rolling z-score detector on forecast residuals."""
import math
from collections import deque


class ResidualAnomalyDetector:
    def __init__(self, window: int = 72, threshold: float = 3.5, min_points: int = 24):
        self.residuals = deque(maxlen=window)
        self.threshold = threshold
        self.min_points = min_points

    def score(self, actual: float, predicted: float):
        """Returns (z_score, is_anomaly). Anomalous residuals are not added to the
        baseline so a spike does not inflate the variance and mask the next one."""
        r = actual - predicted
        if len(self.residuals) < self.min_points:
            self.residuals.append(r)
            return 0.0, False
        mean = sum(self.residuals) / len(self.residuals)
        var = sum((x - mean) ** 2 for x in self.residuals) / (len(self.residuals) - 1)
        z = (r - mean) / max(math.sqrt(var), 1e-9)
        is_anomaly = abs(z) > self.threshold
        if not is_anomaly:
            self.residuals.append(r)
        return z, is_anomaly
