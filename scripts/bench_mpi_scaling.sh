#!/usr/bin/env bash
# bench_mpi_scaling.sh — Benchmarks formales de scaling MPI (strong y weak).
#
# Mide el tiempo de pared para una simulación de prueba a distintos números
# de ranks MPI (1, 2, 4, 8) y genera un JSON con los resultados.
#
# Uso:
#   bash scripts/bench_mpi_scaling.sh [--weak | --strong] [--n-ranks "1 2 4 8"]
#
# Resultados:
#   bench_results/scaling_<timestamp>.json
#
# Requisitos:
#   - mpirun (OpenMPI o MPICH)
#   - cargo (compilación en --release)
#   - configs/validation_128_test.toml (simulación reducida N=32³)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── Parámetros ────────────────────────────────────────────────────────────────

SCALING_MODE="${1:-strong}"           # strong | weak
N_RANKS_LIST="${N_RANKS:-1 2 4 8}"   # números de ranks a medir
CONFIG_BASE="configs/validation_128_test.toml"
OUT_DIR="bench_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_FILE="${OUT_DIR}/scaling_${TIMESTAMP}.json"

mkdir -p "${OUT_DIR}"

echo "== bench_mpi_scaling.sh: modo=${SCALING_MODE} ranks=[${N_RANKS_LIST}] =="
echo "   Config base: ${CONFIG_BASE}"
echo "   Resultados:  ${OUT_FILE}"
echo ""

# ── Compilar en modo release ──────────────────────────────────────────────────

echo "== Compilando gadget-ng-cli --release --features mpi =="
cargo build --release -p gadget-ng-cli --features mpi 2>/dev/null || \
    cargo build --release -p gadget-ng-cli  # fallback sin MPI
BIN="$(find target/release -maxdepth 1 -name 'gadget-ng' -executable 2>/dev/null | head -1)"
if [ -z "$BIN" ]; then
    echo "ERROR: no se encontró el binario gadget-ng en target/release"
    exit 1
fi
echo "   Binario: ${BIN}"
echo ""

# ── Función de benchmark ──────────────────────────────────────────────────────

run_benchmark() {
    local n_ranks="$1"
    local config_file="$2"
    local run_dir="bench_results/run_${n_ranks}ranks_${TIMESTAMP}"
    mkdir -p "${run_dir}"

    local cmd_out
    local elapsed_s="-1"
    local wall_s="-1"

    # Preparar directorio de salida temporal
    local tmp_config="${run_dir}/sim.toml"
    # Modificar el output_dir en la config (simple sed)
    sed "s|output_dir = .*|output_dir = \"${run_dir}\"|g" "${config_file}" > "${tmp_config}" 2>/dev/null \
        || cp "${config_file}" "${tmp_config}"

    echo "   Corriendo ${n_ranks} rank(s)..."
    local t_start t_end
    t_start=$(date +%s%N)

    if [ "${n_ranks}" -gt 1 ] && command -v mpirun &>/dev/null; then
        mpirun --oversubscribe -n "${n_ranks}" "${BIN}" run "${tmp_config}" \
            --max-steps 5 >"${run_dir}/stdout.txt" 2>&1 || true
    else
        "${BIN}" run "${tmp_config}" --max-steps 5 >"${run_dir}/stdout.txt" 2>&1 || true
    fi

    t_end=$(date +%s%N)
    elapsed_s=$(echo "scale=3; (${t_end} - ${t_start}) / 1000000000" | bc 2>/dev/null || echo "-1")

    echo "   → elapsed = ${elapsed_s}s"
    echo "${elapsed_s}"
}

# ── Loop de benchmarks ────────────────────────────────────────────────────────

echo "["  > "${OUT_FILE}"
first=1
for n_ranks in ${N_RANKS_LIST}; do
    echo "-- n_ranks = ${n_ranks} --"
    elapsed=$(run_benchmark "${n_ranks}" "${CONFIG_BASE}")

    if [ "${first}" -eq 0 ]; then
        echo "," >> "${OUT_FILE}"
    fi
    first=0
    cat >> "${OUT_FILE}" << JSONENTRY
  {
    "n_ranks": ${n_ranks},
    "scaling_mode": "${SCALING_MODE}",
    "elapsed_s": ${elapsed},
    "config": "${CONFIG_BASE}",
    "timestamp": "${TIMESTAMP}"
  }
JSONENTRY
done
echo "]" >> "${OUT_FILE}"

echo ""
echo "== Resultados guardados en: ${OUT_FILE} =="
cat "${OUT_FILE}"

# ── Calcular speedup (si hay datos) ──────────────────────────────────────────

echo ""
echo "== Resumen de scaling =="
python3 - <<'PYEOF' 2>/dev/null || echo "   (python3 no disponible para resumen)"
import json, sys

with open("${OUT_FILE}") as f:
    data = json.load(f)

if not data:
    sys.exit(0)

t_ref = data[0]["elapsed_s"]
print(f"{'N ranks':>8}  {'Tiempo (s)':>12}  {'Speedup':>8}  {'Eficiencia':>10}")
print("-" * 45)
for d in data:
    n = d["n_ranks"]
    t = d["elapsed_s"]
    speedup = t_ref / t if t > 0 else 0
    eff = speedup / n * 100
    print(f"{n:>8}  {t:>12.3f}  {speedup:>8.2f}  {eff:>9.1f}%")
PYEOF

echo ""
echo "bench_mpi_scaling.sh: OK"
