#!/usr/bin/env bash
# run_collapse.sh — Ejecuta el experimento de colapso gravitacional frío.
#
# Uso:
#   cd experiments/nbody/cold_collapse
#   bash scripts/run_collapse.sh [--release]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../../../" && pwd)"

RELEASE_FLAG=""
PROFILE="debug"
for arg in "$@"; do
    if [[ "$arg" == "--release" ]]; then
        RELEASE_FLAG="--release"
        PROFILE="release"
    fi
done

echo "=== Experimento: Colapso Gravitacional Frío ==="
echo ""

echo "--- Compilando (${PROFILE}) ---"
cargo build -p gadget-ng-cli ${RELEASE_FLAG} --manifest-path "${REPO_ROOT}/Cargo.toml" 2>&1
BINARY="${REPO_ROOT}/target/${PROFILE}/gadget-ng"

mkdir -p "${EXPERIMENT_DIR}/runs/collapse"

echo "--- Ejecutando simulación de colapso ---"
echo "    N=200, R=1, G=1, T_ff≈2.221, t_total≈5·T_ff"
"${BINARY}" stepping \
    --config "${EXPERIMENT_DIR}/config/collapse.toml" \
    --out "${EXPERIMENT_DIR}/runs/collapse" \
    --snapshot \
    2>&1

N_FRAMES=$(ls "${EXPERIMENT_DIR}/runs/collapse/frames/" 2>/dev/null | wc -l)
echo "    Snapshots guardados: ${N_FRAMES}"

echo ""
echo "=== Listo ==="
echo "Siguiente: python scripts/analyze_collapse.py"
