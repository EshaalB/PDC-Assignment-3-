# CS5 - Sample sort (MPI)

Sorts N = 10^8 32-bit keys spread over p processes, no process holds all of them.

1. local sort - each rank qsorts its N/p keys
2. sampling - each rank takes os*p-1 evenly spaced samples (p-1 if os=1), MPI_Gather to
   rank 0, rank 0 sorts them and picks p-1 splitters, MPI_Bcast
3. partition - cut the sorted keys at the splitters (binary search)
4. exchange - MPI_Alltoall for the counts, MPI_Alltoallv for the keys
5. merge - merge the p sorted runs with a min-heap

Files:
- `sort_common.h` - key generator (key g only depends on seed and g, so every rank can
  make its own part and the data is the same for any p) + checksums
- `serial.c` - baseline, qsort of all N keys
- `parallel.c` - the sample sort, each step timed separately
- `pingpong.c` - measures alpha (latency) and beta (time per byte) for T = alpha + beta*m
- `run_experiments.sh`, `analyze.py` - experiments -> `results/`

Inputs (`-d`): 0 = uniform, 1 = zipf-like (P(j) ~ 1/j for j = 1..10^6, about 5% of keys
are 1), 2 = dup16 (only 16 different values).

## Build

```
make
```

## Run

```
./serial [N] [dist] [seed]                 # defaults 100000000 0 12345
mpirun -np 4 ./parallel -n 100000000 -d 0 -o 1 -c
mpirun -np 2 ./pingpong
```

Options for `parallel`:
- `-n N` number of keys (10^8)
- `-d dist` 0/1/2 (0)
- `-s seed` (12345)
- `-o os` oversampling, os*p-1 samples per rank (1)
- `-c` print how many keys each rank has at the end

## Checking correctness

`parallel` prints `VERIFY ... PASS` if every rank's keys are sorted, the last key of
rank i <= first key of rank i+1, and the total is still N.

It also prints `sum` and `hash` (hash = sum of key*(2*position+1), so it depends on the
order too). These have to be the same as what `serial` prints for the same N/dist/seed,
which means the parallel output is exactly the sorted array.

```
make verify NP=4 N=100000000
```

## Getting the report numbers

```
./run_experiments.sh      # about 10 min and 2 GB RAM. smaller machine: MAXP=4 N=50000000 ./run_experiments.sh
python3 analyze.py
```

N = 10^8, p = 1, 2, 4, 8, all three inputs. Oversampling test: os = 1, 4, 16, 64 at p = 8.
