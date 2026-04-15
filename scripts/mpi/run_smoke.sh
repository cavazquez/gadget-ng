#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

cargo build --features mpi
CFG="${1:-$ROOT/experiments/nbody/mvp_smoke/config/default.toml}"
OUT="${2:-$ROOT/experiments/nbody/mvp_smoke/runs/mpi_smoke}"
rm -rf "$OUT"
mkdir -p "$OUT"

MPIRUN="${MPIRUN:-mpiexec}"
$MPIRUN -n 4 target/debug/gadget-ng stepping --config "$CFG" --out "$OUT" --snapshot

echo "MPI smoke: wrote diagnostics under $OUT"
