#!/usr/bin/env bash
# run_convergence.sh — Ejecuta el experimento de convergencia Kepler.
#
# Corre gadget-ng con 5 valores de dt y guarda los resultados en runs/.
#
# Uso:
#   cd experiments/nbody/two_body_convergence
#   bash scripts/run_convergence.sh [--release]
#
# Requiere:
#   cargo build (--release opcional)
#   gadget-ng CLI en target/debug/gadget-ng o target/release/gadget-ng

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../../../" && pwd)"

# Modo de compilación
RELEASE_FLAG=""
PROFILE="debug"
for arg in "$@"; do
    if [[ "$arg" == "--release" ]]; then
        RELEASE_FLAG="--release"
        PROFILE="release"
    fi
done

echo "=== Experimento: Convergencia Kepler (dos cuerpos) ==="
echo "Repositorio: ${REPO_ROOT}"
echo "Perfil:      ${PROFILE}"
echo ""

# Compilar
echo "--- Compilando gadget-ng (${PROFILE}) ---"
cargo build -p gadget-ng-cli ${RELEASE_FLAG} --manifest-path "${REPO_ROOT}/Cargo.toml" 2>&1
BINARY="${REPO_ROOT}/target/${PROFILE}/gadget-ng"

if [[ ! -f "${BINARY}" ]]; then
    echo "ERROR: No se encontró el binario en ${BINARY}"
    exit 1
fi

# Crear directorio de resultados
mkdir -p "${EXPERIMENT_DIR}/runs"

# Configuraciones a ejecutar
declare -A CONFIGS
CONFIGS["dt020"]="config/kepler_dt020.toml"
CONFIGS["dt050"]="config/kepler_dt050.toml"
CONFIGS["dt100"]="config/kepler_dt100.toml"
CONFIGS["dt200"]="config/kepler_dt200.toml"
CONFIGS["dt500"]="config/kepler_dt500.toml"

# Ejecutar cada configuración
for label in dt020 dt050 dt100 dt200 dt500; do
    config="${EXPERIMENT_DIR}/${CONFIGS[$label]}"
    out_dir="${EXPERIMENT_DIR}/runs/${label}"
    
    echo "--- Ejecutando ${label} → ${out_dir} ---"
    mkdir -p "${out_dir}"
    
    "${BINARY}" stepping \
        --config "${config}" \
        --out "${out_dir}" \
        --snapshot \
        2>&1 | tail -5
    
    echo "    Listo: $(ls "${out_dir}/frames/" 2>/dev/null | wc -l) frames guardados"
done

echo ""
echo "=== Todas las ejecuciones completadas ==="
echo "Siguiente paso: python scripts/analyze_convergence.py"
