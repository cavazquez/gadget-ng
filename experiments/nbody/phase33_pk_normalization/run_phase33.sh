#!/usr/bin/env bash
# run_phase33.sh — orquestador completo del análisis de normalización.
#
# Genera P(k) por seed (via Rust test ignored), calcula estadísticas del
# ensemble, compara A_obs vs A_pred y produce las figuras.
#
# Debe ejecutarse desde la raíz del workspace:
#   bash experiments/nbody/phase33_pk_normalization/run_phase33.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

EXP_DIR="experiments/nbody/phase33_pk_normalization"
OUT_DIR="$EXP_DIR/output"
FIG_DIR="$EXP_DIR/figures"
SCRIPTS_DIR="$EXP_DIR/scripts"
PK_DATA_DIR="$OUT_DIR/pk_data"
PHASE31_SCRIPTS="experiments/nbody/phase31_ensemble_higher_res/scripts"

mkdir -p "$OUT_DIR" "$FIG_DIR" "$PK_DATA_DIR"

log() { echo -e "\n[phase33] $*"; }

log "1. Generando P(k) por seed (Rust, release)..."
cargo test -p gadget-ng-physics --test phase33_pk_normalization --release \
    -- --ignored dump_pk_jsons_for_phase33 --nocapture > "$OUT_DIR/rust_dump.log" 2>&1
echo "  log → $OUT_DIR/rust_dump.log"

log "2. Derivación analítica (tabla de A_pred)..."
python3 "$SCRIPTS_DIR/derive_normalization.py" \
    --output "$OUT_DIR/derivation_table.json" | tee "$OUT_DIR/derivation.log" > /dev/null
echo "  tabla → $OUT_DIR/derivation_table.json"

log "3. Estadísticas de ensemble (N=16³, 6 seeds)..."
python3 "$PHASE31_SCRIPTS/compute_ensemble_stats.py" \
    --pk-files "$PK_DATA_DIR"/pk_N16_seed042.json \
               "$PK_DATA_DIR"/pk_N16_seed137.json \
               "$PK_DATA_DIR"/pk_N16_seed271.json \
               "$PK_DATA_DIR"/pk_N16_seed314.json \
               "$PK_DATA_DIR"/pk_N16_seed512.json \
               "$PK_DATA_DIR"/pk_N16_seed999.json \
    --label "N16_phase33" \
    --output "$OUT_DIR/stats_N16.json" > "$OUT_DIR/stats_N16.log"

log "4. Estadísticas de ensemble (N=32³, 6 seeds)..."
python3 "$PHASE31_SCRIPTS/compute_ensemble_stats.py" \
    --pk-files "$PK_DATA_DIR"/pk_N32_seed042.json \
               "$PK_DATA_DIR"/pk_N32_seed137.json \
               "$PK_DATA_DIR"/pk_N32_seed271.json \
               "$PK_DATA_DIR"/pk_N32_seed314.json \
               "$PK_DATA_DIR"/pk_N32_seed512.json \
               "$PK_DATA_DIR"/pk_N32_seed999.json \
    --label "N32_phase33" \
    --output "$OUT_DIR/stats_N32.json" > "$OUT_DIR/stats_N32.log"

log "5. Comparación A_obs vs A_pred..."
python3 "$SCRIPTS_DIR/measure_and_compare.py" \
    --stats-files "$OUT_DIR/stats_N16.json" "$OUT_DIR/stats_N32.json" \
    --output "$OUT_DIR/A_obs_vs_pred.json" | tee "$OUT_DIR/comparison.log"

log "6. Figuras (residuos y efecto CIC)..."
python3 "$SCRIPTS_DIR/plot_residuals.py" \
    --stats-file "$OUT_DIR/stats_N32.json" \
    --grid-size 32 \
    --out-dir "$FIG_DIR"
python3 "$SCRIPTS_DIR/plot_residuals.py" \
    --stats-file "$OUT_DIR/stats_N16.json" \
    --grid-size 16 \
    --out-dir "$FIG_DIR"

log "Fase 33 completada. Resultados en $OUT_DIR  y figuras en $FIG_DIR"
