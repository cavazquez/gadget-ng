#!/usr/bin/env bash
# ── Benchmarks de validación — Fase 19: PM Distribuido ────────────────────────
#
# Ejecuta el suite de benchmarks para comparar:
#   - PM distribuido (Fase 19: allreduce O(nm³)) vs PM clásico (Fase 18: allgather O(N·P))
#   - Serial (P=1) vs MPI (P=2, P=4) para el path distribuido
#   - Diferentes tamaños de N y grid
#
# Requiere:
#   - gadget-ng compilado: cargo build --release --workspace --features mpi
#   - mpirun disponible (para runs con P>1)
#
# Salidas: results/ (un directorio por configuración/rank)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BIN="${REPO_ROOT}/target/release/gadget-ng"
RESULTS="${SCRIPT_DIR}/results"

if [[ ! -f "${BIN}" ]]; then
    echo "[ERROR] Binario no encontrado: ${BIN}"
    echo "Compila con: cargo build --release --workspace"
    exit 1
fi

mkdir -p "${RESULTS}"

run_serial() {
    local cfg="$1"
    local tag="$2"
    local out="${RESULTS}/${tag}_P1"
    mkdir -p "${out}"
    echo "[RUN] Serial P=1: ${tag}"
    time "${BIN}" --config "${SCRIPT_DIR}/configs/${cfg}" --out-dir "${out}" 2>&1 | tee "${out}/run.log"
}

run_mpi() {
    local cfg="$1"
    local tag="$2"
    local nranks="$3"
    local out="${RESULTS}/${tag}_P${nranks}"
    mkdir -p "${out}"
    echo "[RUN] MPI P=${nranks}: ${tag}"
    time mpirun -n "${nranks}" "${BIN}" --config "${SCRIPT_DIR}/configs/${cfg}" --out-dir "${out}" 2>&1 | tee "${out}/run.log"
}

# ── Bloque A: PM distribuido vs PM clásico (equivalencia física) ──────────────
echo "=== Bloque A: PM Distribuido vs PM Clásico (N=512, EdS) ==="
run_serial "eds_N512_pm_classic.toml"    "N512_classic"
run_serial "eds_N512_pm_dist.toml"       "N512_dist"

# ── Bloque B: PM distribuido en diferentes P ──────────────────────────────────
if command -v mpirun &>/dev/null; then
    echo "=== Bloque B: PM Distribuido — P=1,2,4 (N=512, EdS) ==="
    run_serial "eds_N512_pm_dist.toml"   "N512_dist"
    run_mpi    "eds_N512_pm_dist.toml"   "N512_dist" 2
    run_mpi    "eds_N512_pm_dist.toml"   "N512_dist" 4
else
    echo "[SKIP] mpirun no disponible; omitiendo Bloque B."
fi

# ── Bloque C: N=2000 ΛCDM ─────────────────────────────────────────────────────
echo "=== Bloque C: ΛCDM N=2000 ==="
run_serial "lcdm_N2000_pm_dist.toml"    "N2000_lcdm_dist"
if command -v mpirun &>/dev/null; then
    run_mpi "lcdm_N2000_pm_dist.toml"   "N2000_lcdm_dist" 2
    run_mpi "lcdm_N2000_pm_dist.toml"   "N2000_lcdm_dist" 4
fi

# ── Bloque D: N=4000 EdS (máximo N del benchmark) ────────────────────────────
echo "=== Bloque D: EdS N=4000 ==="
run_serial "eds_N4000_pm_dist.toml"     "N4000_dist"
if command -v mpirun &>/dev/null; then
    run_mpi "eds_N4000_pm_dist.toml"    "N4000_dist" 2
    run_mpi "eds_N4000_pm_dist.toml"    "N4000_dist" 4
fi

echo ""
echo "=== Benchmarks completados. Resultados en: ${RESULTS} ==="
echo "Analizar con: python3 analyze_phase19.py"
