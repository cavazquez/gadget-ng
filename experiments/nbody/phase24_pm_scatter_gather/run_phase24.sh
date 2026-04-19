#!/usr/bin/env bash
# ── Fase 24: PM Scatter/Gather — Script de ejecución ─────────────────────────
#
# Ejecuta comparación Fase 23 (clone+migrate) vs Fase 24 (scatter/gather PM).
#
# Uso:
#   ./run_phase24.sh [--release]
#
# Resultados en: experiments/nbody/phase24_pm_scatter_gather/results/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RESULTS="${SCRIPT_DIR}/results"
CONFIGS="${SCRIPT_DIR}/configs"

RELEASE_FLAG=""
if [[ "${1:-}" == "--release" ]]; then
    RELEASE_FLAG="--release"
    echo "[fase24] Compilando en modo release..."
    cargo build ${RELEASE_FLAG} -p gadget-ng 2>&1 | tail -3
fi

BINARY="${REPO_ROOT}/target/${RELEASE_FLAG:+release}${RELEASE_FLAG:+/}${RELEASE_FLAG:-debug}/gadget-ng"
if [[ ! -f "$BINARY" ]]; then
    echo "[fase24] Compilando gadget-ng..."
    cargo build ${RELEASE_FLAG} -p gadget-ng 2>&1 | tail -3
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Fase 24: PM Scatter/Gather — Benchmark comparativo"
echo "═══════════════════════════════════════════════════════════════"

run_sim() {
    local name="$1"
    local config="$2"
    local outdir="${RESULTS}/${name}"
    mkdir -p "${outdir}"
    echo ""
    echo "▶ Ejecutando: ${name}"
    echo "  Config: ${config}"
    echo "  Salida: ${outdir}"
    time "${BINARY}" stepping --config "${config}" --out "${outdir}" 2>&1 | tail -5
    echo "  ✓ Completado"
}

# ── Baseline: Fase 23 clone+migrate ──────────────────────────────────────────
run_sim "fase23_clone_N512_p1"  "${CONFIGS}/eds_N512_fase23_clone_p1.toml"

# ── Fase 24: scatter/gather PM ───────────────────────────────────────────────
run_sim "fase24_sg_N512_p1"     "${CONFIGS}/eds_N512_fase24_sg_p1.toml"
run_sim "fase24_sg_N1000_p1"    "${CONFIGS}/eds_N1000_fase24_sg_p1.toml"
run_sim "fase24_sg_N2000_lcdm"  "${CONFIGS}/lcdm_N2000_fase24_sg_p1.toml"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Generando análisis comparativo..."
echo "═══════════════════════════════════════════════════════════════"
python3 "${SCRIPT_DIR}/scripts/compare_results.py" "${RESULTS}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Fase 24 completada. Resultados en: ${RESULTS}/"
echo "═══════════════════════════════════════════════════════════════"
