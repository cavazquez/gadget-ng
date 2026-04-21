#!/usr/bin/env bash
# Phase 40 — Orquesta la campaña completa de validación de la convención
# física de normalización de ICs (legacy vs z0_sigma8).
#
# 1. Corre los 7 tests Rust (`cargo test --release --test phase40_…`) →
#    escribe `target/phase40/*.json`.
# 2. Ejecuta un snapshot IC end-to-end con la CLI para cada modo
#    (validación de que la migración TOML funciona sobre el pipeline real).
# 3. Genera las 6 figuras con `plot_phase40_comparison.py`.
# 4. Copia figuras y CSV a `docs/reports/figures/phase40/`.
#
# Uso: `bash experiments/nbody/phase40_physical_ics_normalization/run_phase40.sh`

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PHASE_DIR="$REPO_ROOT/experiments/nbody/phase40_physical_ics_normalization"
OUTPUTS="$PHASE_DIR/outputs"
FIG_DIR="$REPO_ROOT/docs/reports/figures/phase40"
mkdir -p "$OUTPUTS" "$FIG_DIR"

cd "$REPO_ROOT"

echo "=== [phase40] 1/4  Cargando tests Rust ==="
cargo test --release --test phase40_physical_ics_normalization -- --test-threads=1

echo "=== [phase40] 2/4  Snapshot IC end-to-end por modo (CLI) ==="
BIN="$REPO_ROOT/target/release/gadget-ng"
for mode in legacy z0_sigma8; do
    cfg="$PHASE_DIR/configs/lcdm_N32_2lpt_pm_${mode}.toml"
    snap="$OUTPUTS/ic_${mode}.bin"
    "$BIN" snapshot --config "$cfg" --out "$snap" >/dev/null
    echo "   → $snap"
done

echo "=== [phase40] 3/4  Generando figuras ==="
PY="$(command -v python3)"
"$PY" "$PHASE_DIR/scripts/plot_phase40_comparison.py" \
    --matrix "$REPO_ROOT/target/phase40/per_snapshot_metrics.json" \
    --out "$FIG_DIR"

echo "=== [phase40] 4/4  Resumen ==="
echo "Tests JSON : $REPO_ROOT/target/phase40/"
echo "Figuras    : $FIG_DIR/"
ls -1 "$FIG_DIR"
echo "Done."
