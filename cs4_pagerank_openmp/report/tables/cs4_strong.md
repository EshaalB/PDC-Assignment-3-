**CS4 strong scaling (R-MAT scale 20: n = 1,048,576, m = 16,084,838 edges, 24 iterations); serial baseline 1.4175 s, median of 3 runs**

| algorithm | p | T_p (s) | S(p) | E(p) |
|---|---|---|---|---|
| pull | 1 | 1.554 | 0.91 | 0.91 |
| pull | 4 | 0.743 | 1.91 | 0.48 |
| pull | 8 | 0.776 | 1.83 | 0.23 |
| push | 1 | 4.323 | 0.33 | 0.33 |
| push | 4 | 3.031 | 0.47 | 0.12 |
| pull (dynamic,256) | 8 | 0.348 | 4.08 | 0.51 |

*Only the thread counts that were measured are listed. Speedup is relative to the serial program, not to the parallel program on one thread.*
