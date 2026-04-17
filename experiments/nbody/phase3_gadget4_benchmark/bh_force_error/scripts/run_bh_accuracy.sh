#!/usr/bin/env bash
# run_bh_accuracy.sh — Ejecuta el benchmark de precisión BH vs Direct.
#
# Uso:
#   cd <repo_root>
#   bash experiments/nbody/phase3_gadget4_benchmark/bh_force_error/scripts/run_bh_accuracy.sh
#
# Genera:
#   experiments/nbody/phase3_gadget4_benchmark/bh_force_error/results/bh_accuracy.csv
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

echo "[phase3/bh_force_error] Compilando y ejecutando benchmark BH vs Direct..."
cargo test -p gadget-ng-physics \
    --test bh_force_accuracy \
    --release \
    -- --nocapture bh_force_accuracy_full_sweep

echo ""
echo "[phase3/bh_force_error] CSV generado en:"
echo "  experiments/nbody/phase3_gadget4_benchmark/bh_force_error/results/bh_accuracy.csv"
echo ""
echo "Para generar plots:"
echo "  python3 experiments/nbody/phase3_gadget4_benchmark/bh_force_error/scripts/plot_bh_accuracy.py"
