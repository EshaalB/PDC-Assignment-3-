**CS3 reproducibility observations**

| run | threads | mode | hex-float price | observation |
|---|---|---|---|---|
| serial, M=4M | 1 | per-thread serial stream | 0x1.4e26e7b5ac647p+3 | Matches parallel p=1 |
| parallel, M=4M | 1 | perthread | 0x1.4e26e7b5ac647p+3 | Bit-identical to serial |
| parallel, M=4M | 1 | blocks | 0x1.4e44bc1954d83p+3 | Same blocks result as p=3 and p=7 |
| parallel, M=4M | 3 | blocks | 0x1.4e44bc1954d83p+3 | Bit-identical across thread counts |
| parallel, M=4M | 7 | blocks | 0x1.4e44bc1954d83p+3 | Bit-identical across thread counts |
| parallel, M=2M | 4 | shared | varies between repetitions | Not reproducible (reported NO) |
| parallel, M=2M | 4 | sameseed | 0x1.4dd3eb261a4d3p+3 | Repeatable, but streams are correlated |

*The blocks mode is reproducible across p = 1, 3, and 7 because each fixed block owns a deterministic RNG stream and results are combined in a fixed order. Per-thread mode assigns streams according to p, so changing p changes the samples and can change the final estimate. Floating-point addition is also non-associative.*
