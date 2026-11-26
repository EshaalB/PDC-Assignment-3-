**CS3 fixed time budget (-m timed -T 5, seed = 42, Intel Core i7-1255U)**

| p | paths M | Mpaths/s | MC price | SE | |MC - BS| | error / SE |
|---|---|---|---|---|---|---|
| 1 | 318,636,032 | 63.72 | 10.450395 | 0.000825 | 0.000188 | 0.23 |
| 2 | 544,604,160 | 108.91 | 10.449965 | 0.000631 | 0.000618 | 0.98 |
| 4 | 875,233,280 | 175.01 | 10.450146 | 0.000498 | 0.000438 | 0.88 |
| 8 | 1,519,124,480 | 303.71 | 10.450380 | 0.000378 | 0.000204 | 0.54 |

*Increasing threads from 1 to 8 raises simulated paths 4.77x in 5 s and lowers SE by 2.18x (matching sqrt(4.77)).*
