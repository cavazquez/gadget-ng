#!/usr/bin/env bash
# Paridad serial vs MPI usando `barnes_hut.toml` (Barnes–Hut).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="${1:-$ROOT/experiments/nbody/mvp_smoke/config/barnes_hut.toml}"
OUT_BASE="${2:-$ROOT/experiments/nbody/mvp_smoke/runs/parity_cmp_bh}"
exec "$ROOT/scripts/validation/compare_serial_mpi.sh" "$CFG" "$OUT_BASE"
