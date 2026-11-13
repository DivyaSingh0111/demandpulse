"""Additive Holt-Winters (triple exponential smoothing), implemented from scratch."""
from itertools import product


def mape(actual, predicted) -> float:
    pairs = [(a, p) for a, p in zip(actual, predicted) if a > 0]
    return 100.0 * sum(abs(a - p) / a for a, p in pairs) / max(len(pairs), 1)


class HoltWinters:
    def __init__(self, season: int = 24, alpha: float = 0.3, beta: float = 0.01, gamma: float = 0.2):
        self.m, self.alpha, self.beta, self.gamma = season, alpha, beta, gamma
        self.level = self.trend = 0.0
        self.seasonal = []
        self.n = 0

    def fit(self, y):
        """Fit on series y; returns in-sample one-step-ahead predictions."""
        m = self.m
        if len(y) < 2 * m:
            raise ValueError(f"need at least {2 * m} points, got {len(y)}")
        first = sum(y[:m]) / m
        second = sum(y[m:2 * m]) / m
        self.level, self.trend = first, (second - first) / m
        self.seasonal = [v - first for v in y[:m]]
        self.n = 0
        return [self.update(v) for v in y]

    def update(self, value: float) -> float:
        """Consume one observation. Returns the prediction made *before* seeing it."""
        idx = self.n % self.m
        s = self.seasonal[idx]
        pred = self.level + self.trend + s
        new_level = self.alpha * (value - s) + (1 - self.alpha) * (self.level + self.trend)
        self.trend = self.beta * (new_level - self.level) + (1 - self.beta) * self.trend
        self.seasonal[idx] = self.gamma * (value - new_level) + (1 - self.gamma) * s
        self.level = new_level
        self.n += 1
        return pred

    def forecast(self, h: int):
        return [max(0.0, self.level + i * self.trend + self.seasonal[(self.n + i - 1) % self.m])
                for i in range(1, h + 1)]

    @classmethod
    def auto_fit(cls, y, season: int = 24):
        """Grid-search smoothing parameters by in-sample one-step MAPE."""
        best, best_err = None, float("inf")
        for a, b, g in product([0.1, 0.3, 0.5], [0.0, 0.01, 0.05], [0.1, 0.3, 0.5]):
            model = cls(season, a, b, g)
            preds = model.fit(y)
            err = mape(y[2 * season:], preds[2 * season:])
            if err < best_err:
                best, best_err = model, err
        return best


def seasonal_naive(history, h: int, season: int = 24):
    """Baseline: value at hour X equals the value one season ago."""
    return [history[-season + (i % season)] for i in range(h)]


class RobustHoltWinters(HoltWinters):
    """Holt-Winters whose fit/update clip outliers (> k * running MAE) to the forecast,
    so injected spikes in history don't leak into level/seasonal state."""

    def __init__(self, *args, k: float = 4.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.k, self.mae = k, None

    def fit(self, y):
        self.mae = None
        return super().fit(y)

    def update(self, value: float) -> float:
        pred = self.level + self.trend + self.seasonal[self.n % self.m] if self.seasonal else value
        err = abs(value - pred)
        if self.mae is not None and self.n >= 2 * self.m and err > self.k * self.mae:
            value = pred
        else:
            self.mae = err if self.mae is None else 0.95 * self.mae + 0.05 * err
        return super().update(value)


class DemandForecaster:
    """Double seasonality: multiplicative day-of-week factors on top of a robust
    daily (m=24) Holt-Winters fitted to the day-of-week-adjusted series."""

    def __init__(self, start_hour: int = 0):
        self.start = start_hour
        self.dow = [1.0] * 7
        self.hw = None
        self.t = start_hour

    @staticmethod
    def _day(hour: int) -> int:
        return (hour // 24) % 7

    def fit(self, y, start_hour: int = 0):
        self.start = self.t = start_hour
        totals, counts = [0.0] * 7, [0] * 7
        for i, v in enumerate(y):
            d = self._day(start_hour + i)
            totals[d] += v
            counts[d] += 1
        means = [totals[d] / counts[d] if counts[d] else 0 for d in range(7)]
        overall = sum(y) / len(y)
        self.dow = [m / overall if m else 1.0 for m in means]
        adj = [v / self.dow[self._day(start_hour + i)] for i, v in enumerate(y)]
        best, best_err = None, float("inf")
        for a, b, g in product([0.05, 0.1, 0.2, 0.3], [0.0, 0.01], [0.05, 0.1, 0.2]):
            m = RobustHoltWinters(24, a, b, g)
            preds = m.fit(adj)
            err = mape(adj[48:], preds[48:])
            if err < best_err:
                best, best_err = m, err
        self.hw = best
        self.t = start_hour + len(y)
        return self

    def forecast(self, h: int):
        base = self.hw.forecast(h)
        return [base[i] * self.dow[self._day(self.t + i)] for i in range(h)]

    def update(self, value: float):
        f = self.dow[self._day(self.t)]
        self.hw.update(value / f)
        self.t += 1
