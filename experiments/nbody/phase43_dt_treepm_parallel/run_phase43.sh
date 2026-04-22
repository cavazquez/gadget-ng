#!/usr/bin/env bash
# Phase 43 — Orquesta el barrido de `dt` sobre TreePM + softening físico y
# la medición de speedup paralelo.
#
# 1. Corre los 7 tests Rust (`phase43_dt_treepm_parallel.rs`) → escribe
#    `target/phase43/*.json`.
# 2. Genera figuras + CSV con los scripts Python.
# 3. Copia artefactos a `docs/reports/figures/phase43/`.
#
# Variables de entorno útiles:
#   PHASE43_USE_CACHE=1   re-usa `target/phase43/per_snapshot_metrics.json`
#                         sin re-correr la matriz física.
#   PHASE43_QUICK=1       smoke test con `dt ∈ {4e-4, 2e-4}` + adaptive.
#   PHASE43_N=64          sube a N=64³ (coste ~8× del smoke a N=32³).
#   PHASE43_DT5E5=1       añade `dt = 5·10⁻⁵` al barrido.
#   PHASE43_THREADS="1,4,8"  define los hilos Rayon para el test de speedup.
#   PHASE43_SKIP_ADAPTIVE=1  omite la variante adaptativa.
#
# Uso: `bash experiments/nbody/phase43_dt_treepm_parallel/run_phase43.sh`

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PHASE_DIR="$REPO_ROOT/experiments/nbody/phase43_dt_treepm_parallel"
OUTPUTS="$PHASE_DIR/outputs"
FIG_LOCAL="$PHASE_DIR/figures"
FIG_DIR="$REPO_ROOT/docs/reports/figures/phase43"
mkdir -p "$OUTPUTS" "$FIG_LOCAL" "$FIG_DIR"

cd "$REPO_ROOT"

echo "=== [phase43] 1/3  Tests Rust (barrido dt + adaptive + paralelismo) ==="
if [[ "${PHASE43_QUICK:-0}" == "1" ]]; then
    echo "   PHASE43_QUICK=1 → dt ∈ {4e-4, 2e-4} + adaptive (smoke)"
fi
if [[ -n "${PHASE43_N:-}" ]]; then
    echo "   PHASE43_N=${PHASE43_N}"
fi
if [[ "${PHASE43_USE_CACHE:-0}" == "1" ]]; then
    echo "   PHASE43_USE_CACHE=1 → releyendo matriz si existe"
fi

cargo test --release --test phase43_dt_treepm_parallel -- --test-threads=1 --nocapture

echo "=== [phase43] 2/3  Figuras + CSV ==="
PY="$(command -v python3)"
"$PY" "$PHASE_DIR/scripts/plot_dt_effect.py" \
    --matrix "$REPO_ROOT/target/phase43/per_snapshot_metrics.json" \
    --outdir "$FIG_LOCAL"
"$PY" "$PHASE_DIR/scripts/plot_parallel_speedup.py" \
    --test5 "$REPO_ROOT/target/phase43/test5_parallel_speedup.json" \
    --outdir "$FIG_LOCAL"
"$PY" "$PHASE_DIR/scripts/analyze_growth_phase43.py" \
    --matrix "$REPO_ROOT/target/phase43/per_snapshot_metrics.json" \
    --test3 "$REPO_ROOT/target/phase43/test3_adaptive_vs_fixed.json" \
    --outdir "$FIG_LOCAL"

echo "   Copiando figuras/CSV → $FIG_DIR"
cp -f "$FIG_LOCAL"/*.png "$FIG_DIR"/ 2>/dev/null || true
cp -f "$FIG_LOCAL"/*.csv "$FIG_DIR"/ 2>/dev/null || true

echo "=== [phase43] 3/3  Resumen ==="
echo "Tests JSON : $REPO_ROOT/target/phase43/"
echo "Figuras    : $FIG_DIR/"
ls -1 "$FIG_DIR" 2>/dev/null || true
echo "Done."
