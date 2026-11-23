# CS1 - Heat diffusion (MPI)

N x N plate updated for T steps, each cell moves towards the average of its 4 neighbours:

```
new[i][j] = old[i][j] + 0.1 * (up + down + left + right - 4*old[i][j])
```

Edges are fixed (top 100, left 50, right/bottom 0), everything else starts at 0.

Files:
- `serial.c` - baseline, two grids (cur/next) swapped every step
- `parallel.c` - MPI version. 1D strips or 2D tiles (MPI_Cart_create), halo exchange
  with MPI_Sendrecv (blocking) or Isend/Irecv (non-blocking, updates the inside of the
  block while waiting), MPI_Allreduce(MAX) every K steps for convergence
- `run_experiments.sh` - runs all the experiments (3 runs each)
- `analyze.py` - medians -> `results/*.csv` and `results/*.png`

## Build

```
make
make clean
```

## Run

```
./serial [N] [T] [K]          # defaults 1024 500 50
mpirun -np 4 ./parallel -n 1024 -t 500 -k 50 -d 2 -m 1
```

Options for `parallel`:
- `-n N` grid size (1024)
- `-t T` time steps (500)
- `-k K` allreduce every K steps (50)
- `-d 1|2` 1D strips or 2D tiles (2)
- `-m 0|1` blocking or non-blocking (1)
- `-e eps` stop early when max change < eps, 0 = off (0)
- `-r file` compare the final grid with the grid saved by serial

It prints one RESULT line. `time` is the slowest rank, `comm_max` is time spent in halo
exchange, `allreduce` is time in the convergence check, `checksum` is the sum of all cells.

## Checking correctness

```
make verify NP=4 N=1024
```

Runs serial and saves the grid, then runs all 4 MPI versions (1D/2D x blocking/non-blocking),
gathers the grid on rank 0 and compares every cell (tolerance 1e-12). We always got an error
of exactly 0 since the same operations happen in the same order.

## Getting the report numbers

```
./run_experiments.sh      # about 10 min. on a 4 core machine use MAXP=4 ./run_experiments.sh
python3 analyze.py        # needs matplotlib
```

N = 1024 and 4096, T = 500, K = 50, p = 1, 2, 4, 8.
Weak scaling: 2048x2048 cells per process, so N = 2048*sqrt(p). Convergence check test: K = 1 vs 50.
