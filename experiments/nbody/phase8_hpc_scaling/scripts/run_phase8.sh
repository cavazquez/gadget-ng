#!/usr/bin/env bash
# experiments/nbody/phase8_hpc_scaling/scripts/run_phase8.sh
#
# Ejecuta todos los benchmarks de Fase 8 (strong/weak scaling).
#
# Uso:
#   ./run_phase8.sh [--ranks "1 2 4 8"] [--mode allgather|sfc_let|both] [--only-strong|--only-weak]
#
# Requiere:
#   - cargo build --release (o --features mpi para MPI)
#   - mpirun disponible (para P > 1)
#   - gadget-ng compilado

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PHASE_DIR}/../../.." && pwd)"

CONFIG_DIR="${PHASE_DIR}/config"
RESULTS_DIR="${PHASE_DIR}/results"
GADGET_BIN="${REPO_ROOT}/target/release/gadget-ng"

# Defaults.
RANKS="1 2 4 8"
MODE="both"
STRONG=true
WEAK=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --ranks)   RANKS="$2"; shift 2 ;;
        --mode)    MODE="$2"; shift 2 ;;
        --only-strong) WEAK=false; shift ;;
        --only-weak)   STRONG=false; shift ;;
        *) echo "Opción desconocida: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "${RESULTS_DIR}"

# Compilar en release.
echo "=== Compilando gadget-ng (release) ==="
cd "${REPO_ROOT}"
cargo build --release 2>&1 | tail -3

if [[ ! -x "${GADGET_BIN}" ]]; then
    echo "ERROR: binario no encontrado en ${GADGET_BIN}" >&2
    exit 1
fi

run_sim() {
    local config="$1"
    local ranks="$2"
    local tag
    tag=$(basename "${config}" .toml)
    local out_dir="${RESULTS_DIR}/${tag}"
    mkdir -p "${out_dir}"

    local log="${out_dir}/run.log"
    echo "  Ejecutando ${tag} (P=${ranks})..."

    local t_start=$SECONDS
    if [[ "${ranks}" -gt 1 ]]; then
        mpirun --oversubscribe -n "${ranks}" "${GADGET_BIN}" \
            --config "${config}" --out "${out_dir}" > "${log}" 2>&1
    else
        "${GADGET_BIN}" \
            --config "${config}" --out "${out_dir}" > "${log}" 2>&1
    fi
    local t_end=$SECONDS
    local elapsed=$((t_end - t_start))

    echo "    → ${elapsed}s wall, resultados en ${out_dir}"
    echo "${elapsed}" > "${out_dir}/wall_seconds.txt"
}

echo ""
echo "=== Benchmarks Fase 8 ==="

for config in "${CONFIG_DIR}"/*.toml; do
    fname=$(basename "${config}" .toml)

    # Filtrar por modo.
    if [[ "${MODE}" == "allgather" ]] && [[ "${fname}" != *allgather* ]]; then
        continue
    fi
    if [[ "${MODE}" == "sfc_let" ]] && [[ "${fname}" != *sfc_let* ]]; then
        continue
    fi

    # Filtrar por tipo (strong/weak).
    is_strong=false
    is_weak=false
    [[ "${fname}" == strong_* ]] && is_strong=true
    [[ "${fname}" == weak_* ]]   && is_weak=true

    if [[ "${is_strong}" == "true" ]] && [[ "${STRONG}" == "false" ]]; then continue; fi
    if [[ "${is_weak}" == "true" ]]   && [[ "${WEAK}" == "false" ]];   then continue; fi

    # Extraer número de rangos del nombre del archivo (R1, R2, R4, R8).
    ranks=$(echo "${fname}" | grep -oP 'R\K[0-9]+' | tail -1)
    if [[ -z "${ranks}" ]]; then
        ranks=1
    fi

    # Solo ejecutar si el número de rangos está en la lista.
    if ! echo "${RANKS}" | grep -qw "${ranks}"; then
        continue
    fi

    run_sim "${config}" "${ranks}"
done

echo ""
echo "=== Todos los benchmarks completados ==="
echo "Resultados en: ${RESULTS_DIR}"
