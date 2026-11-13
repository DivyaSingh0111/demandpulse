"""Synthetic order-stream generator with daily/weekly seasonality, trend and injected anomalies."""
import math
import random
from dataclasses import dataclass

# Base hourly order volume per product category.
CATEGORIES = {"electronics": 120, "grocery": 300, "fashion": 180, "home": 90}
REGIONS = ["north", "south", "east", "west"]
REGION_WEIGHTS = [0.30, 0.25, 0.20, 0.25]


@dataclass
class Order:
    hour: int
    category: str
    region: str
    quantity: int
    price: float


class DemandSimulator:
    """Each call to step() emits all orders for one simulated hour."""

    def __init__(self, seed: int = 42, anomaly_rate: float = 0.02):
        self.rng = random.Random(seed)
        self.anomaly_rate = anomaly_rate
        self.hour = 0

    @staticmethod
    def expected(category: str, hour: int) -> float:
        base = CATEGORIES[category]
        h, dow = hour % 24, (hour // 24) % 7
        daily = 1 + 0.5 * math.sin(2 * math.pi * (h - 9) / 24)  # peaks mid-afternoon
        weekly = 1.15 if dow >= 5 else 1.0  # weekend uplift
        trend = 1 + 0.0005 * hour  # slow growth
        return base * daily * weekly * trend

    def step(self):
        hour = self.hour
        self.hour += 1
        orders, injected = [], {}
        for cat in CATEGORIES:
            lam = self.expected(cat, hour)
            if self.rng.random() < self.anomaly_rate:
                factor = self.rng.choice([2.5, 0.3])  # demand spike or outage-like drop
                lam *= factor
                injected[cat] = factor
            for _ in range(self._poisson(lam)):
                orders.append(Order(
                    hour=hour,
                    category=cat,
                    region=self.rng.choices(REGIONS, REGION_WEIGHTS)[0],
                    quantity=self.rng.randint(1, 3),
                    price=round(self.rng.uniform(5, 200), 2),
                ))
        return hour, orders, injected

    def _poisson(self, lam: float) -> int:
        if lam > 50:  # normal approximation is accurate and fast here
            return max(0, int(round(self.rng.gauss(lam, math.sqrt(lam)))))
        limit, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= self.rng.random()
            if p < limit:
                return k
            k += 1
