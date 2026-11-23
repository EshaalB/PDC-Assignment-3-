# CS2 - k-mer counting (OpenMP)

Counts how many times every substring of length k appears in a random DNA string of
L = 10^8 bases. Each base is 2 bits (A=0 C=1 G=2 T=3) so a k-mer is just a number from
0 to 4^k - 1 and counting is `hist[code]++`. There are L-k+1 k-mers in total so all the
counts have to add up to that.

Files:
- `kmer_common.h` - DNA generator (fixed seed), 2 bit encoding, histogram checksum
- `serial.c` - baseline
- `parallel.c` - versions A (race, wrong on purpose), B (omp atomic), C (private
  histogram per thread, padded to 64 bytes, merged at the end), U (private but not
  padded, for the false sharing test)
- `run_experiments.sh`, `analyze.py` - experiments -> `results/`

Splitting: the start positions 0..L-k are cut into blocks. A k-mer at the end of a block
reads a few letters of the next block (only reading), but every start position is in
exactly one block so each k-mer is counted once.

## Build

```
make
```

## Run

```
./serial [L] [k] [skew] [seed] [reps]          # defaults 100000000 12 0 42 3
OMP_NUM_THREADS=8 ./parallel -L 100000000 -k 12 -v ABC
```

Options for `parallel`:
- `-L` length (10^8)
- `-k` k-mer length 1..15 (12)
- `-s` 0 = uniform, 1 = skewed (A 45%, C 5%, G 5%, T 45%) (0)
- `-S` seed (42)
- `-r` repeats, median time is printed (3)
- `-v` which versions, any of A B C U (ABC)

Each RESULT line has total, expected, lost, mismatched_bins, extra_MB (memory used by
the private copies) and CORRECT/WRONG.

## Checking correctness

`parallel` first does a serial count and then compares every version with it bin by bin.
A should print WRONG (lost updates), B, C and U should print CORRECT.
(With small inputs A sometimes gets lucky and still prints CORRECT.)

```
make verify L=100000000 K=12 THREADS=8
```

## Getting the report numbers

```
./run_experiments.sh      # about 5 min, MAXT=4 on a 4 core machine
python3 analyze.py
```

L = 10^8, k = 8 and 12, uniform and skewed input, 1/2/4/8 threads.
False sharing test: k = 1 and 2 with U vs C.
