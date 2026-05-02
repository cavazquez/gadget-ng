#!/usr/bin/env bash
# run_bh_accuracy.sh — Ejecuta el benchmark de precisión BH vs Direct.
#
# Uso:
#   cd <repo_root>
#   bash experiments/nbody/phase3_gadget4_benchmark/bh_force_error/scripts/run_bh_accuracy.sh
#
# Genera:
#   results/bh_accuracy.csv            — MAC geométrico (θ), histórico Fase 3
#   results/bh_accuracy_relative.csv   — MAC relativo (ErrTolForceAcc = 0.0025)
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

echo "[phase3/bh_force_error] Compilando y ejecutando benchmark BH vs Direct (geométrico)..."
cargo test -p gadget-ng-physics \
    --test bh_force_accuracy \
    --release \
    -- --nocapture bh_force_accuracy_full_sweep

echo ""
echo "[phase3/bh_force_error] MAC relativo (ErrTolForceAcc)..."
cargo test -p gadget-ng-physics \
    --test bh_force_accuracy \
    --release \
    -- --nocapture bh_force_accuracy_relative_full_sweep

echo ""
echo "[phase3/bh_force_error] CSV en:"
echo "  experiments/nbody/phase3_gadget4_benchmark/bh_force_error/results/bh_accuracy.csv"
echo "  experiments/nbody/phase3_gadget4_benchmark/bh_force_error/results/bh_accuracy_relative.csv"
echo ""
echo "Plots (usa solo el CSV geométrico por defecto):"
echo "  python3 experiments/nbody/phase3_gadget4_benchmark/bh_force_error/scripts/plot_bh_accuracy.py"
