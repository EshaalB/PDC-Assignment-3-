# Case Study 3 - Monte Carlo Option Pricing (OpenMP)

Prices a European call option under geometric Brownian motion with `M >= 1e8` random paths, reports
the standard error, and compares against the Black-Scholes closed form.

**Model:** OpenMP (shared memory). The work is embarrassingly parallel - paths are independent, no data is
shared, and only two doubles (sum and sum of squares) are reduced - so one multicore node with threads
is the natural fit; MPI would add process start-up and a `MPI_Reduce` for no benefit on one machine.

## Option parameters (in `mc_common.h`)

| S0  | K   | r    | sigma | T     | Black-Scholes price |
|-----|-----|------|-------|-------|---------------------|
| 100 | 100 | 0.05 | 0.20  | 1 year | 10.45058357         |

## Files

| File | Purpose |
|------|---------|
| `mc_common.h` | PCG32 generator, SplitMix seeding, Box-Muller, payoff kernel, Black-Scholes, timing helpers (shared by both programs) |
| `serial.c` | Serial baseline (single stream, seed 42) |
| `parallel.c` | OpenMP version with 5 modes (see below) |
| `Makefile` | `make` builds `serial` and `parallel` with `-O3 -march=native`; `make clean` |
| `run_experiments.sh` | Runs every experiment, writes CSV files into `results/` |
| `results/` | Created by the script (input of `../make_graphs.py`) |

## Build

```bash
make                     # gcc -O3 -march=native -fopenmp
```
Requires gcc with OpenMP (Linux / macOS with `brew install gcc` / WSL).

## Run

```bash
./serial   -n 100000000 -s 42 -r 3            # serial baseline, median of 3 runs
./parallel -n 100000000 -t 8  -s 42 -r 3      # 8 threads, per-thread streams (default mode)
./parallel -n 100000000 -t 8  -m blocks       # result bit-identical for every thread count
./parallel -n 2000000   -t 8  -m shared       # BAD on purpose: one generator + lock
./parallel -n 2000000   -t 8  -m sameseed     # BAD on purpose: same seed in every thread
./parallel -t 8 -m timed -T 5                 # fixed 5-second budget, report accuracy reached
```

| Option | Meaning | Default |
|--------|---------|---------|
| `-n M` | number of paths (rounded down to even) | 100000000 |
| `-t p` | threads (parallel only) | `OMP_NUM_THREADS` / all cores |
| `-s seed` | seed | 42 |
| `-r reps` | repetitions; the **median** time is reported | 3 |
| `-m mode` | `perthread`, `blocks`, `timed`, `shared`, `sameseed` | `perthread` |
| `-B blocks` | number of fixed blocks in `blocks` mode | 4096 |
| `-T sec` | time budget for `timed` mode | 5 |
| `--csv` | print one machine-readable `CSV,...` line | off |

## Modes

| Mode | RNG | Combination of partial sums | Reproducible? |
|------|-----|-----------------------------|---------------|
| `perthread` | thread *t* owns stream *t+1* and a contiguous chunk | fixed order thread 0..p-1 | same `(seed, p)` -> bit-identical |
| `blocks` | block *b* owns stream *b+1* (4096 blocks, any thread may run any block) | fixed order block 0..B-1 | **same seed -> bit-identical for every p** |
| `timed` | as `perthread` | as `perthread` | no (amount of work depends on time) |
| `shared` | one global PCG32 behind `omp critical` | fixed order | no, and serialised |
| `sameseed` | identical stream in every thread | fixed order | yes, but statistically wrong |

Design decisions that matter:

* **Independent streams.** Each worker seeds its own PCG32 (`seed`, `stream`) through SplitMix64 so neighbouring stream ids do not start in correlated states.
* **Deterministic reduction.** `omp reduction` combines partial sums in an unspecified order, so the last bits can change between runs. Per-thread results go to cache-line-padded slots (64 B, avoids false sharing) and are added in a fixed order.
* **Box-Muller uses both outputs** (`cos` and `sin`), halving the RNG/`log`/`sqrt` work compared with discarding one.
* `parallel -t 1` executes exactly the same instructions in the same order as `serial`, so their results are bit-identical (this is the correctness check; see `results/cs3_verify.txt`).

## Verification

```bash
./serial -n 4000000 --csv | awk -F, '{print $NF}'
./parallel -n 4000000 -t 1 --csv | awk -F, '{print $NF}'   # identical hex-float
```
For `p > 1` the estimate differs from the serial one by statistical noise only; check that
`|MC - BS| / SE` is below about 3 (the `z` column of the CSV). Floating-point tolerance stated in the report: 4 standard errors.

## Reproduce all experiments and plots

```bash
./run_experiments.sh                  # needs a few minutes; PMAX=8 M=20000000 ./run_experiments.sh for a quick test
cd .. && python3 make_graphs.py --only cs3
```
Figures end up in `report/figures/cs3_*.png`, ready-made tables in `report/tables/cs3_*.md`.
Run on an otherwise idle machine; `OMP_PROC_BIND=close OMP_PLACES=cores` is set by the script.
If you use more threads than physical cores, say so in the report - speedup beyond the core count is not meaningful.

## CSV columns

`tag,mode,threads,M,seed,reps,t_median,t_min,price,se,bs,abs_err,z,repro,hexprice`
