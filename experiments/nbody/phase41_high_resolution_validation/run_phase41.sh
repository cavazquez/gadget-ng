#!/usr/bin/env bash
# Phase 41 — Orquesta la campaña de alta resolución.
#
# 1. Corre los 5 tests Rust (`cargo test --release --test phase41_…`) →
#    escribe `target/phase41/*.json`.
# 2. Ejecuta un snapshot IC end-to-end (CLI) a N=128 en ambos modos para
#    validar que los configs TOML atraviesan el pipeline real.
# 3. Genera las 5 figuras + CSV con `plot_phase41_resolution.py`.
# 4. Copia figuras/CSV a `docs/reports/figures/phase41/`.
#
# Variables de entorno útiles:
#   PHASE41_SKIP_N256=1     saltea N=256 (útil en máquinas con poca RAM)
#   PHASE41_SKIP_CLI=1      saltea los snapshots CLI sanity-check
#
# Uso: `bash experiments/nbody/phase41_high_resolution_validation/run_phase41.sh`

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PHASE_DIR="$REPO_ROOT/experiments/nbody/phase41_high_resolution_validation"
OUTPUTS="$PHASE_DIR/outputs"
FIG_DIR="$REPO_ROOT/docs/reports/figures/phase41"
mkdir -p "$OUTPUTS" "$FIG_DIR"

cd "$REPO_ROOT"

echo "=== [phase41] 1/4  Tests Rust (matriz N ∈ {32,64,128,256}, ~15–35 min) ==="
cargo test --release --test phase41_high_resolution_validation -- --test-threads=1 --nocapture

if [[ "${PHASE41_SKIP_CLI:-0}" != "1" ]]; then
    echo "=== [phase41] 2/4  Snapshot IC end-to-end a N=128 por modo (CLI) ==="
    cargo build --release --bin gadget-ng >/dev/null
    BIN="$REPO_ROOT/target/release/gadget-ng"
    for mode in legacy z0_sigma8; do
        cfg="$PHASE_DIR/configs/lcdm_N128_2lpt_pm_${mode}.toml"
        snap="$OUTPUTS/ic_N128_${mode}.bin"
        "$BIN" snapshot --config "$cfg" --out "$snap" >/dev/null
        echo "   → $snap"
    done
else
    echo "=== [phase41] 2/4  Snapshot CLI OMITIDO (PHASE41_SKIP_CLI=1) ==="
fi

echo "=== [phase41] 3/4  Generando figuras + CSV ==="
PY="$(command -v python3)"
"$PY" "$PHASE_DIR/scripts/plot_phase41_resolution.py" \
    --matrix "$REPO_ROOT/target/phase41/per_snapshot_metrics.json" \
    --outdir "$FIG_DIR" \
    --csv "$FIG_DIR/phase41_summary.csv"

echo "=== [phase41] 4/4  Resumen ==="
echo "Tests JSON : $REPO_ROOT/target/phase41/"
echo "Figuras    : $FIG_DIR/"
ls -1 "$FIG_DIR"
echo "Done."
