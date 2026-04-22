#!/usr/bin/env bash
# Phase 54 — Validación cuantitativa D²(a) con G consistente (auto_g=true)
#
# Ejecuta los 5 tests de crecimiento lineal en N=64/128/256 en modo release.
#
# Uso:
#   ./run_phase54.sh            # corre los 3 resolutions
#   PHASE54_SKIP_N256=1 ./run_phase54.sh   # omite N=256 (~8 min)
#
# Los resultados JSON se guardan en target/phase54/snapshots.json

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== Phase 54: D²(a) validation con G_consistent ==="
echo "G_consistent = 3 × Ω_m × H₀² / 8π ≈ 3.76e-4"
echo "N ∈ {64, 128, 256}, BOX=100 Mpc/h, snapshots a ∈ {0.02, 0.05, 0.10, 0.20, 0.33, 0.50}"
echo ""

TIME_START=$(date +%s)

cargo test -p gadget-ng-physics --release \
    --test phase54_growth_factor_validation \
    -- --test-threads=1 --nocapture 2>&1 | tee /tmp/phase54_output.txt

TIME_END=$(date +%s)
ELAPSED=$((TIME_END - TIME_START))
echo ""
echo "=== Phase 54 completado en ${ELAPSED}s ==="

if [ -f target/phase54/snapshots.json ]; then
    echo "Resultados guardados en target/phase54/snapshots.json"
fi
