#!/usr/bin/env bash
# Phase 42 — Orquesta la validación del softening físico vía TreePM.
#
# 1. Corre los 4 tests Rust (`cargo test --release --test phase42_…`) →
#    escribe `target/phase42/*.json`.
# 2. (Opcional) Ejecuta un snapshot IC end-to-end (CLI) a N=128 por variante
#    para validar que los configs TOML atraviesan el pipeline real.
# 3. Genera las 5 figuras + CSV con `plot_phase42_short_range.py`.
# 4. Copia figuras/CSV a `docs/reports/figures/phase42/`.
#
# Variables de entorno útiles:
#   PHASE42_USE_CACHE=1     relee `target/phase42/per_snapshot_metrics.json`
#                           sin re-ejecutar la matriz (rerun sub-segundo).
#   PHASE42_SKIP_CLI=1      omite los snapshots CLI (por default lo salteamos,
#                           para respetar el coste de TreePM serial).
#   PHASE42_QUICK=1         reduce a N=64 para smoke test local (viola
#                           temporalmente "N ≥ 128" del brief — sólo dev).
#
# Uso: `bash experiments/nbody/phase42_tree_short_range/run_phase42.sh`

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PHASE_DIR="$REPO_ROOT/experiments/nbody/phase42_tree_short_range"
OUTPUTS="$PHASE_DIR/outputs"
FIG_LOCAL="$PHASE_DIR/figures"
FIG_DIR="$REPO_ROOT/docs/reports/figures/phase42"
mkdir -p "$OUTPUTS" "$FIG_LOCAL" "$FIG_DIR"

cd "$REPO_ROOT"

echo "=== [phase42] 1/4  Tests Rust (1 PM + 3 TreePM a N=128, ~30–60 min) ==="
if [[ "${PHASE42_QUICK:-0}" == "1" ]]; then
    echo "   PHASE42_QUICK=1 → N=64 (smoke test)"
fi
if [[ "${PHASE42_USE_CACHE:-0}" == "1" ]]; then
    echo "   PHASE42_USE_CACHE=1 → releyendo matriz desde disco si existe"
fi
cargo test --release --test phase42_tree_short_range -- --test-threads=1 --nocapture

if [[ "${PHASE42_SKIP_CLI:-1}" != "1" ]]; then
    echo "=== [phase42] 2/4  Snapshots IC end-to-end por variante (CLI) ==="
    cargo build --release --bin gadget-ng >/dev/null
    BIN="$REPO_ROOT/target/release/gadget-ng"
    for tag in pm_eps0 treepm_eps001 treepm_eps002 treepm_eps005; do
        cfg="$PHASE_DIR/configs/lcdm_N128_${tag}.toml"
        snap="$OUTPUTS/ic_N128_${tag}.bin"
        "$BIN" snapshot --config "$cfg" --out "$snap" >/dev/null
        echo "   → $snap"
    done
else
    echo "=== [phase42] 2/4  Snapshot CLI OMITIDO (default; PHASE42_SKIP_CLI=0 para activar) ==="
fi

echo "=== [phase42] 3/4  Generando figuras + CSV ==="
PY="$(command -v python3)"
"$PY" "$PHASE_DIR/scripts/plot_phase42_short_range.py" \
    --matrix "$REPO_ROOT/target/phase42/per_snapshot_metrics.json" \
    --outdir "$FIG_LOCAL"

echo "   Copiando figuras/CSV → $FIG_DIR"
cp -f "$FIG_LOCAL"/*.png "$FIG_DIR"/ 2>/dev/null || true
cp -f "$FIG_LOCAL"/*.csv "$FIG_DIR"/ 2>/dev/null || true

echo "=== [phase42] 4/4  Resumen ==="
echo "Tests JSON : $REPO_ROOT/target/phase42/"
echo "Figuras    : $FIG_DIR/"
ls -1 "$FIG_DIR"
echo "Done."
