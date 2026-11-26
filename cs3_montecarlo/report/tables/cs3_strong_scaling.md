**CS3 strong scaling (M = 100,000,000, seed = 42, median of 3 runs, Intel Core i7-1255U)**

| mode | p | T (s) | Mpaths/s | S(p) | E(p) |
|---|---|---|---|---|---|
| serial | 1 | 1.6568 | 60.36 | 1.00 | 1.00 |
| perthread | 1 | 1.5908 | 62.86 | 1.04 | 1.04 |
| perthread | 2 | 0.9573 | 104.46 | 1.73 | 0.87 |
| perthread | 4 | 0.7676 | 130.28 | 2.16 | 0.54 |
| perthread | 8 | 0.3837 | 260.65 | 4.32 | 0.54 |

*Measured with OMP_PROC_BIND=close OMP_PLACES=cores on Intel Core i7-1255U (2 P-cores + 8 E-cores). At p = 4 and p = 8, fast P-cores wait for slower E-cores under static equal splitting.*
