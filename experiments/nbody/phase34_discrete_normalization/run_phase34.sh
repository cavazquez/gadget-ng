#!/usr/bin/env bash
# run_phase34.sh — orquestador completo de Phase 34.
#
# Pipeline:
#   1. Compila y ejecuta los tests Rust de Phase 34 (dumpean JSON a target/phase34/).
#   2. Agrega los JSON a experiments/.../output/stage_table.json.
#   3. Genera las 5 figuras obligatorias y las copia a docs/reports/figures/phase34/.
#
# Ejecutar desde la raíz del workspace:
#   bash experiments/nbody/phase34_discrete_normalization/run_phase34.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

EXP_DIR="experiments/nbody/phase34_discrete_normalization"
OUT_DIR="$EXP_DIR/output"
FIG_DIR="$EXP_DIR/figures"
SCRIPTS_DIR="$EXP_DIR/scripts"
DOC_FIG_DIR="docs/reports/figures/phase34"
TEST_DUMP_DIR="target/phase34"

mkdir -p "$OUT_DIR" "$FIG_DIR" "$DOC_FIG_DIR"

log() { echo -e "\n[phase34] $*"; }

log "1/3 · Ejecutando tests Rust (release)..."
cargo test -p gadget-ng-physics --test phase34_discrete_normalization --release \
    -- --test-threads=1 --nocapture > "$OUT_DIR/rust_tests.log" 2>&1 || {
    echo "  tests Rust fallaron — revisar $OUT_DIR/rust_tests.log"
    exit 1
}
echo "  log → $OUT_DIR/rust_tests.log"

log "2/3 · Construyendo tabla por etapa..."
python3 "$SCRIPTS_DIR/stage_table.py" \
    --input "$TEST_DUMP_DIR" \
    --output "$OUT_DIR/stage_table.json" \
    --markdown "$OUT_DIR/stage_table.md"
echo "  tabla → $OUT_DIR/stage_table.json"
echo "  markdown → $OUT_DIR/stage_table.md"

log "3/3 · Generando figuras..."
python3 "$SCRIPTS_DIR/plot_stages.py" \
    --input "$TEST_DUMP_DIR" \
    --fig-dir "$FIG_DIR"
cp -f "$FIG_DIR"/*.png "$DOC_FIG_DIR/"
echo "  figuras → $FIG_DIR/ (copia en $DOC_FIG_DIR/)"

log "Phase 34 completada. Ver docs/reports/2026-04-phase34-*.md para la interpretación."
