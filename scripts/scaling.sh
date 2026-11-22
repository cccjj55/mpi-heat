#!/usr/bin/env bash
# Strong scaling: fixed problem, growing rank count.
# Weak scaling: fixed work per rank, growing problem.
# Both write CSV next to the binary; the README table is generated from these files.
set -euo pipefail

cd "$(dirname "$0")/.."
BIN=build/heat_mpi
mkdir -p results
NP_LIST=${NP_LIST:-1 2 4 8}
MPIRUN=${MPIRUN:-mpirun}

echo "mode,n,steps,procs,seconds,ms_per_step,mlups,checksum" > results/strong.csv
for p in $NP_LIST; do
    $MPIRUN --oversubscribe -np "$p" ./$BIN -n 1024 -t 200 -r 3 | tail -1 >> results/strong.csv
done

echo "mode,n,steps,procs,seconds,ms_per_step,mlups,checksum" > results/weak.csv
for p in $NP_LIST; do
    $MPIRUN --oversubscribe -np "$p" ./$BIN -w 128 -t 200 -r 3 | tail -1 >> results/weak.csv
done

echo "--- strong ---"
cat results/strong.csv
echo "--- weak ---"
cat results/weak.csv
