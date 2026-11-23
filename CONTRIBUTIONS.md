# Contributions - 23L-0648, 23L-0570

| Member | Roll No. | Main responsibility |
|--------|----------|---------------------|
| Eshaal Rehamtullah | 23L-0648 | CS1 (heat, MPI), CS2 (k-mer, OpenMP), CS5 (sample sort, MPI), report |
| Atika Hussain | 23L-0570 | CS3 (Monte Carlo, OpenMP), CS4 (PageRank, OpenMP) |

Hours are rough estimates.

## Eshaal Rehamtullah (23L-0648)

| Task | Approx. hours | Files |
|------|---------------|-------|
| Repo setup, directory structure, top-level Makefile | 1 | `Makefile`, `README.md`, `.gitignore` |
| CS1 serial + MPI (1D/2D, blocking/non-blocking, Allreduce check), verification | 6 | `cs1_heat_mpi/serial.c`, `parallel.c`, `Makefile` |
| CS1 experiments (strong, weak, K=1 vs 50) + graphs | 4 | `cs1_heat_mpi/run_experiments.sh`, `analyze.py`, `results/` |
| CS2 serial + OpenMP versions A/B/C/U, verification | 5 | `cs2_kmer_openmp/*.c`, `kmer_common.h` |
| CS2 experiments (scaling, atomic vs private, false sharing) + graphs | 3 | `cs2_kmer_openmp/run_experiments.sh`, `analyze.py`, `results/` |
| CS5 serial + MPI sample sort, ping-pong, verification | 6 | `cs5_samplesort_mpi/*.c`, `sort_common.h` |
| CS5 experiments (scaling, phases, oversampling, alpha/beta) + graphs | 4 | `cs5_samplesort_mpi/run_experiments.sh`, `analyze.py`, `results/` |
| Final report | 6 | `report/report.docx` |
| **Total** | **about 35** | |

## Atika Hussain (23L-0570)

| Task | Approx. hours | Files |
|------|---------------|-------|
| CS3 serial + parallel Monte Carlo (RNG modes, reproducibility) | 10 | `cs3_montecarlo/*` |
| CS4 serial + pull/push PageRank, schedules | 12 | `cs4_pagerank_openmp/*` |
| CS3/CS4 graphs and tables | 3 | `cs3_montecarlo/report/`, `cs4_pagerank_openmp/report/` |
| **Total** | **about 25** | |
