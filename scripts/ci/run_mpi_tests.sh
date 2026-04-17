#!/usr/bin/env bash
# scripts/ci/run_mpi_tests.sh
#
# Ejecuta los tests MPI del crate gadget-ng-parallel con mpirun en 2 y 4 rangos.
#
# Uso:
#   ./scripts/ci/run_mpi_tests.sh [--ranks N]
#
# Requiere: libopenmpi-dev, openmpi-bin, Rust con feature mpi.
# En CI: el job mpi-multirank en .github/workflows/ci.yml instala las dependencias.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Parseado de argumentos.
RANK_LIST=(2 4)
while [[ $# -gt 0 ]]; do
    case $1 in
        --ranks)
            RANK_LIST=($2)
            shift 2
            ;;
        *)
            echo "Uso: $0 [--ranks 'N1 N2 ...']" >&2
            exit 1
            ;;
    esac
done

echo "=== gadget-ng MPI tests ==="
echo "Repositorio: ${REPO_ROOT}"
echo "Rangos a probar: ${RANK_LIST[*]}"
echo ""

# Compilar tests MPI sin ejecutar.
echo "--- Compilando tests MPI ---"
cd "${REPO_ROOT}"
cargo test -p gadget-ng-parallel --features mpi --no-run 2>&1

# Localizar el binario de test compilado.
TEST_BIN=$(find target/debug/deps -name "sfc_hardening-*" -executable 2>/dev/null | head -1)
if [[ -z "${TEST_BIN}" ]]; then
    echo "ERROR: no se encontró binario sfc_hardening. Asegúrate de compilar con --features mpi." >&2
    exit 1
fi

LET_BIN=$(find target/debug/deps -name "let_validation-*" -executable 2>/dev/null | head -1)
DTREE_BIN=$(find target/debug/deps -name "distributed_tree_energy-*" -executable 2>/dev/null | head -1)

PASS=0
FAIL=0

run_test() {
    local bin="$1"
    local name="$2"
    local ranks="$3"
    echo ""
    echo "--- ${name} con ${ranks} rangos ---"
    if mpirun --oversubscribe -n "${ranks}" "${bin}" --test-threads=1 2>&1; then
        echo "[OK] ${name} @ ${ranks} rangos"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] ${name} @ ${ranks} rangos" >&2
        FAIL=$((FAIL + 1))
    fi
}

# Ejecutar tests seriales (siempre).
echo ""
echo "--- Tests seriales (sin mpirun) ---"
cargo test -p gadget-ng-parallel --features mpi 2>&1 && PASS=$((PASS + 1)) || FAIL=$((FAIL + 1))

# Ejecutar tests con mpirun para cada número de rangos.
for ranks in "${RANK_LIST[@]}"; do
    if [[ -n "${TEST_BIN}" ]]; then
        run_test "${TEST_BIN}" "sfc_hardening" "${ranks}"
    fi
    if [[ -n "${LET_BIN}" ]]; then
        run_test "${LET_BIN}" "let_validation" "${ranks}"
    fi
    if [[ -n "${DTREE_BIN}" ]]; then
        run_test "${DTREE_BIN}" "distributed_tree_energy" "${ranks}"
    fi
done

echo ""
echo "=== Resumen: ${PASS} OK, ${FAIL} FAIL ==="

if [[ ${FAIL} -gt 0 ]]; then
    exit 1
fi
