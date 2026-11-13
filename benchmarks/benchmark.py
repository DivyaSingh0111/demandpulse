"""Reproducible offline evaluation. Run: python -m benchmarks.benchmark"""
import time
from pathlib import Path

from app.anomaly import ResidualAnomalyDetector
from app.engine import DemandEngine
from app.forecaster import DemandForecaster, mape, seasonal_naive
from app.simulator import CATEGORIES, DemandSimulator

TRAIN, TEST, H = 24 * 28, 24 * 14, 24


def build_series(seed=7, anomaly_rate=0.02):
    sim = DemandSimulator(seed, anomaly_rate)
    series = {c: [] for c in CATEGORIES}
    labels = {c: [] for c in CATEGORIES}
    for _ in range(TRAIN + TEST):
        _, orders, injected = sim.step()
        counts = {c: 0 for c in CATEGORIES}
        for o in orders:
            counts[o.category] += o.quantity
        for c in CATEGORIES:
            series[c].append(counts[c])
            labels[c].append(c in injected)
    return series, labels


def evaluate_forecasts(series, labels):
    rows = []
    for c, y in series.items():
        hw_err, daily_err, weekly_err, n = 0.0, 0.0, 0.0, 0
        # Rolling-origin backtest: re-forecast the next 24h at the start of every test day.
        for origin in range(TRAIN, TRAIN + TEST - H + 1, 24):
            model = DemandForecaster().fit(y[origin - TRAIN:origin], start_hour=origin - TRAIN)
            truth = y[origin:origin + H]
            clean = [i for i in range(H) if not labels[c][origin + i]]  # score on non-anomalous hours
            t = [truth[i] for i in clean]
            hw = model.forecast(H)
            d = seasonal_naive(y[:origin], H, 24)
            w = seasonal_naive(y[:origin], H, 168)
            hw_err += mape(t, [hw[i] for i in clean])
            daily_err += mape(t, [d[i] for i in clean])
            weekly_err += mape(t, [w[i] for i in clean])
            n += 1
        rows.append((c, hw_err / n, daily_err / n, weekly_err / n))
    return rows


def evaluate_anomalies(series, labels):
    tp = fp = fn = 0
    for c, y in series.items():
        model = DemandForecaster().fit(y[:TRAIN], start_hour=0)
        det = ResidualAnomalyDetector()
        for t in range(TRAIN, TRAIN + TEST):
            pred = model.forecast(1)[0]
            _, flag = det.score(y[t], pred)
            model.update(pred if flag else y[t])
            truth = labels[c][t]
            tp += flag and truth
            fp += flag and not truth
            fn += (not flag) and truth
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return tp, fp, fn, precision, recall


def evaluate_throughput(hours=500):
    eng = DemandEngine(seed=11)
    eng.warmup()
    events, t0 = 0, time.perf_counter()
    for _ in range(hours):
        eng.tick()
        events = eng.events_total
    elapsed = time.perf_counter() - t0
    return events, elapsed, events / elapsed


def main():
    series, labels = build_series()
    fc = evaluate_forecasts(series, labels)
    tp, fp, fn, p, r = evaluate_anomalies(series, labels)
    events, secs, eps = evaluate_throughput()

    lines = ["# DemandPulse benchmark results", "",
             f"Train window: {TRAIN} h, test window: {TEST} h, horizon: {H} h (rolling origin, daily).", "",
             "## 24-hour forecast MAPE (non-anomalous hours)", "",
             "| Category | DemandPulse model | Naive (yesterday) | Naive (last week) | Gain vs best naive |",
             "|---|---|---|---|---|"]
    for c, hw, d, w in fc:
        best = min(d, w)
        lines.append(f"| {c} | {hw:.2f}% | {d:.2f}% | {w:.2f}% | {100 * (best - hw) / best:.1f}% |")
    avg = [sum(r_[i] for r_ in fc) / len(fc) for i in (1, 2, 3)]
    best = min(avg[1], avg[2])
    lines.append(f"| **mean** | **{avg[0]:.2f}%** | **{avg[1]:.2f}%** | **{avg[2]:.2f}%** | **{100 * (best - avg[0]) / best:.1f}%** |")
    lines += ["", "## Anomaly detection (vs. injected ground truth)", "",
              f"TP={tp}, FP={fp}, FN={fn} → precision **{p:.2%}**, recall **{r:.2%}**", "",
              "## End-to-end throughput (single process)", "",
              f"{events:,} order events through aggregate→detect→forecast in {secs:.2f}s → **{eps:,.0f} events/sec**", ""]
    out = "\n".join(lines)
    print(out)
    Path(__file__).with_name("results.md").write_text(out)


if __name__ == "__main__":
    main()
