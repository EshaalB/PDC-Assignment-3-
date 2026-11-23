#!/bin/bash
# CS1 - runs every experiment needed for the report.
# Every configuration is run RUNS times; analyze.py takes the median.
# Output: results/raw.txt (one RESULT line per run)
#
# Usage:  ./run_experiments.sh          (takes ~10 minutes on 8 cores)
#         MAXP=4 ./run_experiments.sh   (if your machine has fewer cores)
set -e
cd "$(dirname "$0")"
make -s all

RUNS=${RUNS:-3}
MAXP=${MAXP:-8}                  # do not go above the number of PHYSICAL cores
T=500
OUT=results/raw.txt
mkdir -p results
: > "$OUT"
lscpu > results/machine.txt 2>/dev/null || true

PLIST=""
for p in 1 2 4 8 16; do [ "$p" -le "$MAXP" ] && PLIST="$PLIST $p"; done

run() {   # run <label> <command...>
    local label=$1; shift
    for r in $(seq 1 "$RUNS"); do
        echo "exp=$label run=$r $("$@" | grep RESULT)" | tee -a "$OUT"
    done
}

# 1) serial baseline
for N in 1024 4096; do
    run serial ./serial $N $T 50
done

# 2) strong scaling: fixed N, p = 1,2,4,8 ; 1-D vs 2-D ; blocking vs non-blocking
for N in 1024 4096; do
  for p in $PLIST; do
    for d in 1 2; do
      for m in 0 1; do
        run strong mpirun -np $p ./parallel -n $N -t $T -k 50 -d $d -m $m
      done
    done
  done
done

# 3) weak scaling: the same number of cells per rank (2048 x 2048), so N = 2048 * sqrt(p)
for p in $PLIST; do
    N=$(python3 -c "import math; print(int(round(2048*math.sqrt($p))))")
    for d in 1 2; do
        run weak mpirun -np $p ./parallel -n $N -t $T -k 50 -d $d -m 1
    done
done

# 4) cost of the convergence check: K = 1 vs K = 50 (2-D, non-blocking)
for N in 1024 4096; do
  for p in $PLIST; do
    for K in 1 50; do
        run kcheck mpirun -np $p ./parallel -n $N -t $T -k $K -d 2 -m 1
    done
  done
done

echo "done -> $OUT   (now run: python3 analyze.py)"
