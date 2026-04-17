#!/usr/bin/env bash
# Fase 4: Ejecutar todos los benchmarks de multipolos + softening
# Requiere: cargo build --release completado

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_DIR="$REPO_ROOT/experiments/nbody/phase4_multipole_softening/results"

mkdir -p "$RESULTS_DIR"

echo "=== Fase 4: Benchmarks Multipolos + Softening ==="
echo "Repositorio: $REPO_ROOT"
echo ""

echo "[1/4] Ablación softening multipolar (bare vs softened × concentración)..."
cargo test -p gadget-ng-physics --test bh_force_accuracy --release -- \
    bh_softened_multipoles_ablation --nocapture 2>&1 | tee "$RESULTS_DIR/softened_ablation.log"

echo ""
echo "[2/4] Análisis radial de error (r/ε binning)..."
cargo test -p gadget-ng-physics --test bh_force_accuracy --release -- \
    bh_radial_error_analysis --nocapture 2>&1 | tee "$RESULTS_DIR/radial_analysis.log"

echo ""
echo "[3/4] Barrido de criterio relativo (precision vs costo)..."
cargo test -p gadget-ng-physics --test bh_force_accuracy --release -- \
    bh_relative_criterion_sweep --nocapture 2>&1 | tee "$RESULTS_DIR/criterion_sweep.log"

echo ""
echo "[4/4] Generando plots..."
python3 "$SCRIPT_DIR/plot_phase4.py" "$RESULTS_DIR"

echo ""
echo "=== Fase 4 completada. Resultados en $RESULTS_DIR ==="
