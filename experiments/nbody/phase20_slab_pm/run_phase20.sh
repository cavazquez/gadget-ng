#!/usr/bin/env bash
# ── Phase 20: Slab PM distribuido — Benchmarks de validación ─────────────────
#
# Ejecuta gadget-ng con pm_slab=true para P=1,2,4 y compara con Phase 19.
# Genera diagnostics.jsonl en results/ para cada configuración.
#
# Uso:
#   bash run_phase20.sh
#
# Requiere:
#   - gadget-ng compilado en release (cargo build --release)
#   - mpirun disponible para P>1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/../../../target/release/gadget-ng"
RESULTS="${SCRIPT_DIR}/results"
CONFIGS="${SCRIPT_DIR}/configs"

if [[ ! -f "${BINARY}" ]]; then
    echo "[phase20] Compilando gadget-ng en release..."
    cargo build --release --manifest-path "${SCRIPT_DIR}/../../../Cargo.toml"
fi

mkdir -p "${RESULTS}"

run_case() {
    local name="$1"
    local config="$2"
    local nranks="$3"
    local outdir="${RESULTS}/${name}_P${nranks}"
    mkdir -p "${outdir}"

    echo "[phase20] ${name} P=${nranks} → ${outdir}"
    if [[ "${nranks}" -eq 1 ]]; then
        "${BINARY}" stepping \
            --config "${config}" \
            --out "${outdir}" \
            2>"${outdir}/stderr.log"
    else
        mpirun -n "${nranks}" "${BINARY}" stepping \
            --config "${config}" \
            --out "${outdir}" \
            2>"${outdir}/stderr.log"
    fi
    echo "[phase20] ✓ ${name} P=${nranks}"
}

# ── EdS N=512 grid 16³ ───────────────────────────────────────────────────────
for P in 1 2 4; do
    if [[ $((16 % P)) -eq 0 ]]; then
        run_case "eds_N512_slab" "${CONFIGS}/eds_N512_slab.toml" "${P}"
    fi
done

# Comparativa Phase 19: misma config pero pm_distributed (requiere config separada)
# Usar la config de Phase 19 si existe
P19_CONFIG="${SCRIPT_DIR}/../phase19_distributed_pm/configs/eds_N512_pm_dist.toml"
if [[ -f "${P19_CONFIG}" ]]; then
    for P in 1 2 4; do
        outdir="${RESULTS}/eds_N512_phase19_P${P}"
        mkdir -p "${outdir}"
        echo "[phase20] Phase 19 reference eds_N512 P=${P}"
        if [[ "${P}" -eq 1 ]]; then
            "${BINARY}" stepping --config "${P19_CONFIG}" --out "${outdir}" \
                2>"${outdir}/stderr.log" || true
        else
            mpirun -n "${P}" "${BINARY}" stepping --config "${P19_CONFIG}" --out "${outdir}" \
                2>"${outdir}/stderr.log" || true
        fi
    done
fi

# ── ΛCDM N=2000 grid 32³ ─────────────────────────────────────────────────────
for P in 1 2 4; do
    if [[ $((32 % P)) -eq 0 ]]; then
        run_case "lcdm_N2000_slab" "${CONFIGS}/lcdm_N2000_slab.toml" "${P}"
    fi
done

# ── EdS N=4000 grid 32³ ──────────────────────────────────────────────────────
for P in 1 2 4; do
    if [[ $((32 % P)) -eq 0 ]]; then
        run_case "eds_N4000_slab" "${CONFIGS}/eds_N4000_slab.toml" "${P}"
    fi
done

echo ""
echo "[phase20] Todos los runs completados. Ejecuta analyze_phase20.py para análisis."
