#!/bin/bash
# CS5 - runs every experiment needed for the report.
# Every configuration is run RUNS times; analyze.py takes the median.
# Output: results/raw.txt and results/pingpong.txt
#
# Usage:  ./run_experiments.sh          (takes ~10 minutes, needs ~2 GB RAM)
#         MAXP=4 N=50000000 ./run_experiments.sh   (smaller machine)
set -e
cd "$(dirname "$0")"
make -s all

RUNS=${RUNS:-3}
MAXP=${MAXP:-8}                 # number of PHYSICAL cores
N=${N:-100000000}               # 10^8 keys
OUT=results/raw.txt
mkdir -p results
: > "$OUT"
lscpu > results/machine.txt 2>/dev/null || true

PLIST=""
for p in 1 2 4 8 16; do [ "$p" -le "$MAXP" ] && PLIST="$PLIST $p"; done

run() {   # run <label> <command...>
    local label=$1; shift
    for r in $(seq 1 "$RUNS"); do
        echo "exp=$label run=$r $("$@" | grep -E 'RESULT|VERIFY|COUNTS' | tr '\n' ' ')" | tee -a "$OUT"
    done
}

# 1) ping-pong: latency alpha and per-byte time beta
mpirun -np 2 ./pingpong | tee results/pingpong.txt

# 2) serial baseline (single-process qsort of all N keys)
for d in 0 1 2; do
    run serial ./serial $N $d
done

# 3) strong scaling, all three distributions, oversampling 1
for d in 0 1 2; do
  for p in $PLIST; do
    run strong mpirun -np $p ./parallel -n $N -d $d -o 1 -c
  done
done

# 4) effect of oversampling on load balance (largest p)
for d in 0 1 2; do
  for os in 1 4 16 64; do
    run oversample mpirun -np $MAXP ./parallel -n $N -d $d -o $os -c
  done
done

echo "done -> $OUT   (now run: python3 analyze.py)"
