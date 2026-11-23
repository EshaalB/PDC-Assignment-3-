**CS3 RNG modes (M = 2,000,000, p = 4, seed = 42)**

| RNG mode | time (s) | Mpaths/s | MC price | SE | |MC − BS| | error / SE | repeatable |
|---|---|---|---|---|---|---|---|
| perthread | 0.0408 | 49.01 | 10.43233905 | 0.01039680 | 0.01824453 | 1.75 | yes |
| shared | 0.2601 | 7.69 | 10.44913179 | 0.01040647 | 0.00145178 | 0.14 | NO |
| sameseed | 0.0477 | 41.90 | 10.43211896 | 0.01038903 | 0.01846461 | 1.78 | yes |

*The shared generator is protected by a lock, so its runtime is much higher. The same-seed mode repeats correlated samples and therefore does not provide independent worker streams. A small error in one run does not make that RNG mode statistically valid.*
