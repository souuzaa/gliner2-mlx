## Quality check: fast-decisions (dev split)

| Variant | Size | Accuracy (dev) | Same answer as fp32 | Mean / max abs Δp | Median latency | Peak memory |
|---|---|---|---|---|---|---|
| fp32 reference | 1.9 GB | 63.7% | — | — | 140 ms | 3.01 GB |
| GLiNER2.5-Decide-bf16 | 973 MB | 63.7% | 99.5% | 0.0025 / 0.038 | 110 ms | 1.67 GB |
| GLiNER2.5-Decide-8bit | 567 MB | 63.8% | 99.6% | 0.0037 / 0.056 | 118 ms | 1.37 GB |
| GLiNER2.5-Decide-4bit | 350 MB | 63.7% | 97.2% | 0.0265 / 0.364 | 117 ms | 1.19 GB |

Measured on 1700 rows / 2900 heads of the `fastino/fast-decisions` **dev** split (the published benchmark uses a held-out test split, so these accuracies are not comparable to it). Accuracy is head-level exact match averaged over the 17 domains; multi-label heads use threshold 0.5. Latency is per `predict()` call at batch size 1.
