#!/usr/bin/env bash
# run_block_compare.sh — Ejecuta colapso frío en modo global dt y block timesteps.
#
# Uso:
#   cd <repo_root>
#   bash experiments/nbody/phase3_gadget4_benchmark/block_timestep_compare/scripts/run_block_compare.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

EXP_DIR="experiments/nbody/phase3_gadget4_benchmark/block_timestep_compare"
BIN="./target/release/gadget-ng"

echo "[block_compare] Compilando en release..."
cargo build -p gadget-ng-cli --release 2>&1 | grep -E "Compiling|Finished|error"

echo ""
echo "[block_compare] Ejecutando: timestep global..."
"$BIN" stepping \
    --config "$EXP_DIR/config/global_dt.toml" \
    --out    "$EXP_DIR/runs/global_dt" \
    --snapshot

echo "[block_compare] Ejecutando: block timesteps (hierarchical)..."
"$BIN" stepping \
    --config "$EXP_DIR/config/hierarchical.toml" \
    --out    "$EXP_DIR/runs/hierarchical" \
    --snapshot

echo ""
echo "[block_compare] Runs completados."
echo "  global_dt:    $EXP_DIR/runs/global_dt/"
echo "  hierarchical: $EXP_DIR/runs/hierarchical/"
echo ""
echo "Para analizar:"
echo "  python3 $EXP_DIR/scripts/analyze_block_compare.py"
