#!/bin/bash
# CS2 - runs every experiment needed for the report.
# ./parallel and ./serial already repeat each measurement 3 times (-r 3)
# and print the MEDIAN, so every line here is already a median of 3 runs.
# Output: results/raw.txt
#
# Usage:  ./run_experiments.sh          (takes ~5 minutes)
#         MAXT=4 ./run_experiments.sh   (if your machine has fewer cores)
set -e
cd "$(dirname "$0")"
make -s all

L=${L:-100000000}               # 10^8 bases
MAXT=${MAXT:-8}                 # number of PHYSICAL cores
REPS=3
OUT=results/raw.txt
mkdir -p results
: > "$OUT"
lscpu > results/machine.txt 2>/dev/null || true

TLIST=""
for t in 1 2 4 8 16; do [ "$t" -le "$MAXT" ] && TLIST="$TLIST $t"; done

# 1) serial baseline, k = 8 and 12, uniform and skewed input
for skew in 0 1; do
  for k in 8 12; do
    echo "exp=serial $(./serial $L $k $skew 42 $REPS | grep RESULT)" | tee -a "$OUT"
  done
done

# 2) strong scaling of versions A (race), B (atomic), C (private)
for skew in 0 1; do
  for k in 8 12; do
    for t in $TLIST; do
      OMP_NUM_THREADS=$t ./parallel -L $L -k $k -s $skew -r $REPS -v ABC \
        | grep RESULT | sed 's/^/exp=strong /' | tee -a "$OUT"
    done
  done
done

# 3) false sharing: tiny k, padded (C) vs unpadded (U) private counters
for k in 1 2; do
  for t in $TLIST; do
    OMP_NUM_THREADS=$t ./parallel -L $L -k $k -r $REPS -v UC \
      | grep RESULT | sed 's/^/exp=falseshare /' | tee -a "$OUT"
  done
done

echo "done -> $OUT   (now run: python3 analyze.py)"
