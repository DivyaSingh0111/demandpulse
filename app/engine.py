"""Streaming engine: ingest -> aggregate -> detect -> forecast."""
import time
from collections import Counter, deque
from datetime import datetime, timedelta

from .anomaly import ResidualAnomalyDetector
from .forecaster import DemandForecaster, mape
from .simulator import CATEGORIES, REGIONS, DemandSimulator

HORIZON = 24         # forecast hours ahead
WARMUP_HOURS = 24 * 14
REFIT_EVERY = 24
HISTORY_HOURS = 24 * 28
START = datetime(2026, 1, 1)


class DemandEngine:
    def __init__(self, seed: int = 42, anomaly_rate: float = 0.02):
        self.sim = DemandSimulator(seed, anomaly_rate)
        self.history = {c: deque(maxlen=HISTORY_HOURS) for c in CATEGORIES}
        self.preds = {c: deque(maxlen=HISTORY_HOURS) for c in CATEGORIES}
        self.models = {}
        self.detectors = {c: ResidualAnomalyDetector() for c in CATEGORIES}
        self.anomalies = deque(maxlen=25)
        self.region_counts = Counter()
        self.revenue = 0.0
        self.events_total = 0
        self.tick_ms = deque(maxlen=100)
        self.last_eps = 0.0
        self.hour = -1

    @staticmethod
    def aggregate(orders):
        """Collapse raw order events into per-category and per-region unit demand."""
        by_cat, by_region, revenue = Counter(), Counter(), 0.0
        for o in orders:
            by_cat[o.category] += o.quantity
            by_region[o.region] += o.quantity
            revenue += o.quantity * o.price
        return by_cat, by_region, revenue

    def warmup(self):
        for _ in range(WARMUP_HOURS):
            hour, orders, _ = self.sim.step()
            by_cat, _, _ = self.aggregate(orders)
            for c in CATEGORIES:
                self.history[c].append(by_cat[c])
            self.hour = hour
        self._refit()

    def _refit(self):
        for c in CATEGORIES:
            hist = list(self.history[c])
            self.models[c] = DemandForecaster().fit(hist, start_hour=self.hour + 1 - len(hist))

    def tick(self):
        t0 = time.perf_counter()
        hour, orders, injected = self.sim.step()
        by_cat, by_region, revenue = self.aggregate(orders)
        self.hour = hour
        self.region_counts = by_region
        self.revenue = revenue
        for c in CATEGORIES:
            actual = by_cat[c]
            predicted = self.models[c].forecast(1)[0]
            z, is_anom = self.detectors[c].score(actual, predicted)
            # Robust update: feed the forecast instead of an outlier so spikes don't corrupt state.
            self.models[c].update(predicted if is_anom else actual)
            self.history[c].append(actual)
            self.preds[c].append(predicted)
            if is_anom:
                self.anomalies.appendleft({
                    "time": self._label(hour), "category": c, "actual": actual,
                    "expected": round(predicted), "z": round(z, 2),
                    "type": "spike" if z > 0 else "drop", "injected": c in injected,
                })
        if (hour + 1) % REFIT_EVERY == 0:
            self._refit()
        elapsed = time.perf_counter() - t0
        self.tick_ms.append(elapsed * 1000)
        self.events_total += len(orders)
        self.last_eps = len(orders) / max(elapsed, 1e-9)
        return self.snapshot()

    @staticmethod
    def _label(hour: int) -> str:
        return (START + timedelta(hours=hour)).strftime("%a %d %b %H:00")

    def snapshot(self, window: int = 72):
        cats = {}
        for c in CATEGORIES:
            hist = list(self.history[c])[-window:]
            preds = list(self.preds[c])[-window:]
            live = list(self.history[c])[-len(self.preds[c]):][-168:]
            cats[c] = {
                "actual": hist,
                "predicted": [None] * (len(hist) - len(preds)) + [round(p, 1) for p in preds],
                "forecast": [round(v, 1) for v in self.models[c].forecast(HORIZON)],
                "latest": hist[-1] if hist else 0,
                "live_mape": round(mape(live, list(self.preds[c])[-168:]), 2) if self.preds[c] else None,
            }
        return {
            "hour": self.hour,
            "time": self._label(self.hour),
            "labels": [self._label(h) for h in range(self.hour - window + 1, self.hour + 1)],
            "forecast_labels": [self._label(h) for h in range(self.hour + 1, self.hour + 1 + HORIZON)],
            "categories": cats,
            "regions": {r: self.region_counts.get(r, 0) for r in REGIONS},
            "revenue": round(self.revenue, 2),
            "anomalies": list(self.anomalies),
            "metrics": {
                "events_total": self.events_total,
                "events_per_sec": round(self.last_eps),
                "avg_tick_ms": round(sum(self.tick_ms) / max(len(self.tick_ms), 1), 2),
            },
        }
