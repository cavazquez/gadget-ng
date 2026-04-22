#!/usr/bin/env bash
# Phase 55 — Comparación espectro de masas FoF vs HMF (Press-Schechter / Sheth-Tormen)
#
# Evoluciona a z=0 con G consistente y compara halos FoF con la HMF analítica.
#
# Uso:
#   ./run_phase55.sh            # corre N=64, 128, 256
#   PHASE55_SKIP_N256=1 ./run_phase55.sh   # omite N=256 (~8 min)
#
# Los resultados JSON se guardan en target/phase55/fof_results.json

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== Phase 55: FoF vs HMF — N=64/128/256, BOX=300 Mpc/h ==="
echo "Evolución desde a=0.02 hasta a=1.0 (z=0)"
echo "FoF: b=0.2, min_particles=20"
echo ""

TIME_START=$(date +%s)

cargo test -p gadget-ng-physics --release \
    --test phase55_fof_vs_hmf \
    -- --test-threads=1 --nocapture 2>&1 | tee /tmp/phase55_output.txt

TIME_END=$(date +%s)
ELAPSED=$((TIME_END - TIME_START))
echo ""
echo "=== Phase 55 completado en ${ELAPSED}s ==="

if [ -f target/phase55/fof_results.json ]; then
    echo "Resultados guardados en target/phase55/fof_results.json"
fi
