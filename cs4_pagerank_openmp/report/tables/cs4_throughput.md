**CS4 throughput (median of 3 runs)**

| configuration | time (s) | M edges/s | GB/s (model) |
|---|---|---|---|
| push atomic, p=1 | 4.323 | 89 | 2.21 |
| push plain, p=1 | 1.947 | 198 | 4.90 |
| pull static, p=1 | 1.554 | 248 | 3.76 |
| push atomic, p=4 | 3.031 | 127 | 3.14 |
| pull static, p=4 | 0.743 | 519 | 7.86 |
| pull guided,64, p=8 | 0.878 | 440 | 6.65 |
| pull static, p=8 | 0.776 | 497 | 7.52 |
| pull edge-bal., p=8 | 0.551 | 701 | 10.61 |
| pull dynamic,256, p=8 | 0.348 | 1110 | 16.80 |

*GB/s = (12*m + 48*n bytes per iteration for pull, 20*m + 72*n for push) * iterations / time. It is a model, not a hardware counter reading.*
