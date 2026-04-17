#!/usr/bin/env bash
# run_stability.sh — Ejecuta el experimento de estabilidad de Plummer.
#
# Corre: (1) ejecución serial, (2) MPI 2 rangos, (3) MPI 4 rangos (si disponible).
#
# Uso:
#   cd experiments/nbody/plummer_stability
#   bash scripts/run_stability.sh [--release] [--no-mpi]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../../../" && pwd)"

RELEASE_FLAG=""
PROFILE="debug"
NO_MPI=false

for arg in "$@"; do
    case "$arg" in
        --release) RELEASE_FLAG="--release"; PROFILE="release" ;;
        --no-mpi)  NO_MPI=true ;;
    esac
done

echo "=== Experimento: Estabilidad Esfera de Plummer ==="
echo ""

# Compilar
echo "--- Compilando gadget-ng (${PROFILE}) ---"
cargo build -p gadget-ng-cli ${RELEASE_FLAG} --manifest-path "${REPO_ROOT}/Cargo.toml" 2>&1
BINARY="${REPO_ROOT}/target/${PROFILE}/gadget-ng"

# Compilar con feature MPI si no se solicitó --no-mpi
if [[ "$NO_MPI" == "false" ]]; then
    cargo build -p gadget-ng-cli ${RELEASE_FLAG} \
        --features gadget-ng-parallel/mpi \
        --manifest-path "${REPO_ROOT}/Cargo.toml" 2>&1 || {
        echo "AVISO: compilación MPI falló; usando solo serial"
        NO_MPI=true
    }
    BINARY_MPI="${REPO_ROOT}/target/${PROFILE}/gadget-ng"
fi

mkdir -p "${EXPERIMENT_DIR}/runs"

# 1. Ejecución serial
echo ""
echo "--- [1/3] Serial ---"
OUT_SERIAL="${EXPERIMENT_DIR}/runs/serial"
mkdir -p "${OUT_SERIAL}"
"${BINARY}" stepping \
    --config "${EXPERIMENT_DIR}/config/serial.toml" \
    --out "${OUT_SERIAL}" \
    --snapshot \
    2>&1 | tail -5
echo "    Snapshots: $(ls "${OUT_SERIAL}/frames/" 2>/dev/null | wc -l)"

# 2. MPI 2 rangos
if [[ "$NO_MPI" == "false" ]] && command -v mpirun &>/dev/null; then
    echo ""
    echo "--- [2/3] MPI 2 rangos ---"
    OUT_MPI2="${EXPERIMENT_DIR}/runs/mpi_2rank"
    mkdir -p "${OUT_MPI2}"
    mpirun -n 2 "${BINARY_MPI}" stepping \
        --config "${EXPERIMENT_DIR}/config/mpi_2rank.toml" \
        --out "${OUT_MPI2}" \
        --snapshot \
        2>&1 | tail -5
    echo "    Snapshots: $(ls "${OUT_MPI2}/frames/" 2>/dev/null | wc -l)"

    echo ""
    echo "--- [3/3] MPI 4 rangos ---"
    OUT_MPI4="${EXPERIMENT_DIR}/runs/mpi_4rank"
    mkdir -p "${OUT_MPI4}"
    mpirun -n 4 "${BINARY_MPI}" stepping \
        --config "${EXPERIMENT_DIR}/config/mpi_4rank.toml" \
        --out "${OUT_MPI4}" \
        --snapshot \
        2>&1 | tail -5
    echo "    Snapshots: $(ls "${OUT_MPI4}/frames/" 2>/dev/null | wc -l)"
else
    echo "AVISO: mpirun no disponible o --no-mpi activo; omitiendo ejecuciones MPI"
fi

echo ""
echo "=== Ejecuciones completadas ==="
echo "Siguiente paso: python scripts/analyze_stability.py"
