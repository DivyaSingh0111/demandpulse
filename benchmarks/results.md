# DemandPulse benchmark results

Train window: 672 h, test window: 336 h, horizon: 24 h (rolling origin, daily).

## 24-hour forecast MAPE (non-anomalous hours)

| Category | DemandPulse model | Naive (yesterday) | Naive (last week) | Gain vs best naive |
|---|---|---|---|---|
| electronics | 12.12% | 13.87% | 12.98% | 6.6% |
| grocery | 8.59% | 12.15% | 10.80% | 20.5% |
| fashion | 9.45% | 12.44% | 11.37% | 16.9% |
| home | 10.90% | 14.21% | 14.08% | 22.6% |
| **mean** | **10.26%** | **13.17%** | **12.31%** | **16.6%** |

## Anomaly detection (vs. injected ground truth)

TP=21, FP=0, FN=2 → precision **100.00%**, recall **91.30%**

## End-to-end throughput (single process)

471,762 order events through aggregate→detect→forecast in 2.38s → **198,263 events/sec**
