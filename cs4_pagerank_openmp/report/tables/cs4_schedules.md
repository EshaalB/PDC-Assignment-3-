**CS4 time vs schedule, pull, p = 8, median of 3 runs**

| schedule,chunk | time (s) | speedup vs serial | measured R |
|---|---|---|---|
| dynamic,256 | 0.348 | 4.08 | 1.00* |
| edge-balanced static | 0.551 | 2.57 | 1.18* |
| static | 0.776 | 1.83 | 2.18 |
| guided,64 | 0.878 | 1.61 | 1.23* |

**R of dynamic/guided/edge is a sum over all iterations, so it can look balanced although each iteration waited for a straggler; judge these schedules by time. Only the chunk sizes you actually ran are shown (dynamic,256 and guided,64).*
