**CS4 pull vs push, median of 3 runs**

| variant | threads | time (s) | relative |
|---|---|---|---|
| pull | 1 | 1.554 | 1.00 (reference) |
| push, plain stores (no atomics) | 1 | 1.947 | 1.25x pull |
| push, atomic | 1 | 4.323 | 2.78x pull, 2.22x plain push |
| pull, static | 4 | 0.743 | 1.00 (reference) |
| push, atomic, static | 4 | 3.031 | 4.08x pull |

*Pure cost of the atomic instruction = +122%. Extra slowdown at p = 4 comes from contention on hub vertices.*
