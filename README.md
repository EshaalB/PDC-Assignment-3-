# PDC Assignment 3 — Parallel Case Studies (OpenMP / MPI)

Group: **GroupXX** (replace `XX` with the group number everywhere before submission)

## Directory layout (as required by the assignment)

```
GroupXX_Assignment3/
  cs1_heat_mpi/         serial.c parallel.c Makefile README.md   (MPI)
  cs2_kmer_openmp/      serial.c parallel.c Makefile README.md   (OpenMP)
  cs3_montecarlo/       serial.c parallel.c Makefile README.md   (MPI or OpenMP)
  cs4_pagerank_openmp/  serial.c parallel.c Makefile README.md   (OpenMP)
  cs5_samplesort_mpi/   serial.c parallel.c Makefile README.md   (MPI)
  report/               GroupXX_report.pdf
  slides/               GroupXX_slides.pdf (or .pptx)
  CONTRIBUTIONS.md
```

## Toolchain

- Linux / WSL (Ubuntu), `gcc -O3 -fopenmp`, OpenMPI (`mpicc`, `mpirun`)
- Check: `mpicc --version`, `mpirun --version`, `gcc --version`

## Build everything

```
make            # builds every case study
make clean
```

Each case-study directory has its own `Makefile` and `README.md` with run
commands and input parameters.

## Timing methodology (agreed by both members)

- Serial baseline compiled with the same `-O3` flags as the parallel code.
- Speedup `S(p) = T_serial / T_p`, efficiency `E(p) = S(p) / p`.
- Every point = **median of 3 runs** on an idle machine.
- Do not run more ranks/threads than physical cores without saying so
  (`--oversubscribe`).

## Submission

Zip the repository contents as `GroupXX_Assignment3.zip`
(`make zip` from the root).
