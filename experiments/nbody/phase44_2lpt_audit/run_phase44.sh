#!/usr/bin/env bash
# Phase 44 — auditoría 2LPT: orquesta el test A/B (Fixed vs LegacyBuggy) y
# genera figuras + CSV.
#
# Variables de entorno útiles:
#   PHASE44_N=32        tamaño de la retícula (default 32, pot. 2, [16, 128]).
#   PHASE44_USE_CACHE=1 re-usa target/phase44/per_snapshot_metrics.json.
#
# Uso: bash experiments/nbody/phase44_2lpt_audit/run_phase44.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PHASE_DIR="$REPO_ROOT/experiments/nbody/phase44_2lpt_audit"
OUTPUTS="$PHASE_DIR/outputs"
FIG_LOCAL="$PHASE_DIR/figures"
FIG_DIR="$REPO_ROOT/docs/reports/figures/phase44"
mkdir -p "$OUTPUTS" "$FIG_LOCAL" "$FIG_DIR"

cd "$REPO_ROOT"

echo "=== [phase44] 1/3  Unit tests k-space (ic_2lpt) ==="
cargo test -p gadget-ng-core ic_2lpt 2>&1 | tee "$OUTPUTS/unit_tests.log"

echo "=== [phase44] 2/3  Integration test A/B (N=${PHASE44_N:-32}) ==="
if [[ "${PHASE44_USE_CACHE:-0}" == "1" && -f "$REPO_ROOT/target/phase44/per_snapshot_metrics.json" ]]; then
    echo "   PHASE44_USE_CACHE=1 → releyendo cache existente"
else
    cargo test -p gadget-ng-physics --test phase44_2lpt_audit --release -- \
        --test-threads=1 --nocapture 2>&1 | tee "$OUTPUTS/phase44_test_run.log"
fi

echo "=== [phase44] 3/3  Figuras + CSV ==="
python3 "$PHASE_DIR/scripts/plot_ab_comparison.py"

# Copiar figuras a docs/reports/figures/phase44/
cp -v "$FIG_LOCAL"/*.png "$FIG_DIR/" || true

echo "=== [phase44] done ==="
echo "  JSON : $REPO_ROOT/target/phase44/per_snapshot_metrics.json"
echo "  CSV  : $FIG_LOCAL/phase44_summary.csv"
echo "  PNG  : $FIG_LOCAL/phase44_metrics_vs_a.png"
echo "  doc  : $REPO_ROOT/docs/reports/2026-04-phase44-2lpt-audit-fix.md"
