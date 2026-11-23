# Case Study 4 - Parallel PageRank (OpenMP)

PageRank with damping `d = 0.85` on a CSR graph, iterated until `||r(t+1) - r(t)||_1 < eps` (default `1e-9`).
Compares **pull** and **push** formulations and the OpenMP schedules `static`, `dynamic`, `guided`
(with several chunk sizes) plus an edge-balanced static partition.

**Model:** OpenMP (shared memory): the graph (~200 MB at the default size) fits in one node's memory and every
iteration needs random access to all rank values, which would force all-to-all communication under MPI.

## Files

| File | Purpose |
|------|---------|
| `graph.h` | R-MAT generator, SNAP edge-list loader, CSR construction (out- and in-neighbour CSR, no self-loops/duplicates), degree histogram dump |
| `pr_common.h` | Options, graph loading, timing helpers, **serial reference PageRank** (used by `serial.c` and for verification) |
| `serial.c` | Serial baseline (pull, CSR) |
| `parallel.c` | Parallel pull / push / push-without-atomics, all schedules, built-in verification |
| `Makefile` | `make`, `make clean` (`-O3 -march=native -fopenmp`) |
| `run_experiments.sh` | Runs all experiments, writes CSV into `results/` |

## Build and run

```bash
make
./serial   -scale 20 -ef 16 -r 3                       # baseline
./parallel -t 8 -a pull -s static   -v                 # pull, schedule(static), verify against serial
./parallel -t 8 -a pull -s dynamic -c 256              # schedule(dynamic,256)
./parallel -t 8 -a pull -s guided  -c 64               # schedule(guided,64)
./parallel -t 8 -a pull -s edge                        # static partition balanced by edge count
./parallel -t 8 -a push -s static  -v                  # push with #pragma omp atomic
./parallel -t 1 -a pushplain                           # push loop WITHOUT atomics (1 thread only)
./parallel -t 8 -f web-Google.txt -v                   # real SNAP graph instead of R-MAT
```

| Option | Meaning | Default |
|--------|---------|---------|
| `-scale S -ef F` | R-MAT with `2^S` vertices and `F*2^S` edges (before duplicate removal) | 20, 16 |
| `-perm` | randomly relabel vertices (removes the id/degree correlation of R-MAT) | off |
| `-f file` | SNAP edge list (`u v` per line, `#` comments) overrides R-MAT | - |
| `-seed X` | graph seed (graph is identical for any thread count) | 1 |
| `-t p` | threads | all cores |
| `-a` | `pull`, `push`, `pushplain` | pull |
| `-s`, `-c` | schedule `static|dynamic|guided|edge` and chunk size | static, 0 |
| `-r reps` | repetitions; **median** time reported (graph build is excluded) | 3 |
| `-e eps`, `-i maxit` | convergence threshold / iteration cap | 1e-9, 200 |
| `-v` | verify against the serial result (L1 and max-abs difference) | off |
| `-D prefix` | write degree histograms `prefix_in.csv`, `prefix_out.csv` | off |
| `--csv` | machine-readable output | off |

Default graph (`-scale 20 -ef 16`): n = 1,048,576 vertices, m = 16,084,838 edges after duplicate/self-loop removal (>= 10^6 required),
about 48 % dangling vertices, 24 iterations. For a **real** graph download `web-Google.txt.gz` from
<https://snap.stanford.edu/data/web-Google.html>, un-gzip it, and pass it with `-f` (cite it in the report).

## Algorithms

* **Dangling nodes.** A vertex with no out-edges would leak rank. Its rank is collected each iteration (`D = sum of r(u)` over dangling `u`, an OpenMP reduction) and redistributed uniformly: `r'(v) = (1-d)/n + d*(sum_{u->v} r(u)/outdeg(u) + D/n)`. The ranks therefore always sum to 1 (printed after every run).
* **Pull** - the thread that owns destination `v` reads `contrib[u] = r[u]/outdeg[u]` for every in-neighbour `u`. Random reads, private writes, **no atomics**.
* **Push** - the thread that owns source `u` adds `contrib[u]` to each out-neighbour with `#pragma omp atomic` (conflicting writes). A final pass applies damping and computes the convergence metric.
* **Schedules** use `schedule(runtime)` configured through `omp_set_schedule`, so one binary covers every configuration. `edge` cuts the vertex range so each thread gets about the same number of edges.
* Per-thread loop times are accumulated in cache-line-padded slots and give the **load-imbalance ratio** `R = slowest thread time / mean thread time` (measured on the edge loop, summed over all iterations). `work_R` in the CSV is the machine-independent version computed from edge counts of the static blocks.
* First-touch initialisation is done in parallel (NUMA friendly).

## Verification

`-v` runs the serial implementation and prints `L1 diff` and `max|diff|`. All variants must report **PASS**
(L1 < 1e-7; typical values are 1e-14 .. 1e-13, caused only by the different summation order of floating-point
additions). `run_experiments.sh` verifies pull and push with every schedule and stores the log in `results/cs4_verify.txt`.

## Reproduce all experiments and plots

```bash
./run_experiments.sh                      # strong scaling, schedule sweep, atomics, relabelling, verification
SCALE=18 PMAX=4 ./run_experiments.sh      # smaller/faster
cd .. && python3 make_graphs.py --only cs4 --peak-bw 38.4    # --peak-bw: your DRAM bandwidth in GB/s (optional)
```
Figures: `report/figures/cs4_*.png`; tables: `report/tables/cs4_*.md`. Use `--reps N` in `make_graphs.py` if you changed `REPS`.

## CSV columns

`tag,algo,sched,chunk,threads,n,m,iters,t_median,t_min,edges_per_s,gbps_model,imbalance_R,work_R,verify_L1`

`gbps_model` is an estimate: pull moves about `12*m + 48*n` bytes per iteration, push about `20*m + 72*n` (derivation in the report, Q4).
