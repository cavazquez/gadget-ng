#!/usr/bin/env bash
# =============================================================================
# run_phase30.sh — Fase 30: Validación contra Referencia Externa
# =============================================================================
#
# OBJETIVO: Validar gadget-ng contra la referencia analítica EH + CAMB opcional,
# comparando FORMA espectral, CRECIMIENTO temporal, y 1LPT vs 2LPT.
#
# IMPORTANTE: Esta validación NO compara amplitudes absolutas de P(k) vs P_EH
# porque existe un offset conocido de normalización. Valida FORMA y CRECIMIENTO.
#
# PRERREQUISITOS:
#   - Binario gadget-ng compilado en ../../target/release/gadget-ng
#   - Python 3 con numpy y matplotlib (opcional: camb para referencia externa)
#   - Ejecutar desde el directorio de la fase:
#     cd experiments/nbody/phase30_linear_reference && bash run_phase30.sh
#
# COSMOLOGÍA DE REFERENCIA (Planck 2018):
#   Omega_m=0.315, Omega_b=0.049, h=0.674, n_s=0.965, sigma8=0.8, T_CMB=2.7255K
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

BINARY="${REPO_ROOT}/target/release/gadget-ng"
SCRIPTS="${SCRIPT_DIR}/scripts"
CONFIGS="${SCRIPT_DIR}/configs"
OUTDIR="${SCRIPT_DIR}/output"

BOX_MPC_H=100.0
H_DIMLESS=0.674
A_INIT=0.02

mkdir -p "${OUTDIR}"

# ── Colores para output ───────────────────────────────────────────────────────
RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GRN}[phase30]${NC} $*"; }
warn() { echo -e "${YLW}[AVISO]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Verificar prerrequisitos ──────────────────────────────────────────────────

if [[ ! -x "${BINARY}" ]]; then
    warn "Binario no encontrado: ${BINARY}"
    warn "Compilando en modo release..."
    (cd "${REPO_ROOT}" && cargo build --release --bin gadget-ng 2>&1) || {
        err "No se pudo compilar gadget-ng. Continúa solo con tests unitarios."
        BINARY=""
    }
fi

PYTHON=$(command -v python3 || command -v python || true)
if [[ -z "${PYTHON}" ]]; then
    warn "Python no encontrado — se omiten análisis Python"
fi

# ── Paso 1: Generar referencia EH (+CAMB opcional) ────────────────────────────

log "=== Paso 1: Generar espectro de referencia EH ==="

if [[ -n "${PYTHON}" ]]; then
    "${PYTHON}" "${SCRIPTS}/generate_reference_pk.py" \
        --omega-m 0.315 \
        --omega-b 0.049 \
        --h       "${H_DIMLESS}" \
        --n-s     0.965 \
        --sigma8  0.8 \
        --z       0.0 \
        --k-min   0.01 \
        --k-max   5.0 \
        --n-k     300 \
        --try-camb \
        --out "${OUTDIR}/reference_pk.json" \
    && log "  Referencia EH guardada en output/reference_pk.json" \
    || warn "  No se pudo generar referencia Python"
else
    warn "  Python no disponible — omitiendo referencia EH"
fi

# ── Paso 2: Ejecutar simulaciones ─────────────────────────────────────────────

log "=== Paso 2: Ejecutar simulaciones ==="

if [[ -z "${BINARY}" ]]; then
    warn "  Binario no disponible — omitiendo simulaciones"
else
    declare -A CONFIGS_NAMES=(
        ["lcdm_N32_a002_1lpt_pm"]="N=32³, 1LPT, PM,     a_init=0.02"
        ["lcdm_N32_a002_2lpt_pm"]="N=32³, 2LPT, PM,     a_init=0.02"
        ["lcdm_N32_a002_2lpt_treepm"]="N=32³, 2LPT, TreePM, a_init=0.02"
        ["lcdm_N32_a005_2lpt_pm"]="N=32³, 2LPT, PM,     a_init=0.05"
    )

    for name in "${!CONFIGS_NAMES[@]}"; do
        desc="${CONFIGS_NAMES[$name]}"
        cfg="${CONFIGS}/${name}.toml"
        out="${OUTDIR}/${name}"
        mkdir -p "${out}"

        log "  Simulación: ${desc}"
        "${BINARY}" --config "${cfg}" --output-dir "${out}" \
        && log "    OK: ${out}" \
        || warn "    FALLIDA: ${name}"
    done

    # N=64³ es opcional (requiere más RAM y tiempo)
    if [[ "${RUN_N64:-0}" == "1" ]]; then
        log "  Simulación N=64³ (RUN_N64=1)..."
        mkdir -p "${OUTDIR}/lcdm_N64_a002_2lpt_pm"
        "${BINARY}" \
            --config "${CONFIGS}/lcdm_N64_a002_2lpt_pm.toml" \
            --output-dir "${OUTDIR}/lcdm_N64_a002_2lpt_pm" \
        && log "    OK: N=64³" \
        || warn "    FALLIDA: N=64³"
    else
        warn "  N=64³ omitido (usar RUN_N64=1 para activar)"
    fi
fi

# ── Paso 3: Extraer P(k) de los snapshots ────────────────────────────────────

log "=== Paso 3: Extraer P(k) de snapshots ==="

# gadget-ng exporta snapshots en formato JSON si se compila con --features json-output
# Si no, los snapshots están en formato binario y necesitan post-procesamiento
# Aquí asumimos que el binario exporta P(k) en JSON directamente.
# Ajustar según el formato real de salida de gadget-ng.

for name in lcdm_N32_a002_1lpt_pm lcdm_N32_a002_2lpt_pm; do
    snap_dir="${OUTDIR}/${name}"
    if [[ -d "${snap_dir}" ]]; then
        # El snapshot inicial es snap_0000.json o equivalente
        snap_init=$(ls "${snap_dir}"/snap_0000*.json 2>/dev/null | head -1 || true)
        if [[ -n "${snap_init}" && -f "${snap_init}" ]]; then
            cp "${snap_init}" "${OUTDIR}/pk_init_${name#lcdm_N32_a002_}.json"
            log "  Extraído: pk_init_${name#lcdm_N32_a002_}.json"
        fi
    fi
done

# ── Paso 4: Comparar forma espectral ─────────────────────────────────────────

log "=== Paso 4: Comparar forma espectral vs referencia ==="

if [[ -n "${PYTHON}" && -f "${OUTDIR}/reference_pk.json" ]]; then
    PK_1LPT="${OUTDIR}/pk_init_1lpt_pm.json"
    PK_2LPT="${OUTDIR}/pk_init_2lpt_pm.json"

    if [[ -f "${PK_1LPT}" ]]; then
        ARGS_2LPT=""
        [[ -f "${PK_2LPT}" ]] && ARGS_2LPT="--pk-gadget-2lpt ${PK_2LPT}"

        "${PYTHON}" "${SCRIPTS}/compare_pk_shape.py" \
            --pk-gadget     "${PK_1LPT}" \
            --pk-ref        "${OUTDIR}/reference_pk.json" \
            --box           "${BOX_MPC_H}" \
            --h             "${H_DIMLESS}" \
            ${ARGS_2LPT} \
            --out-prefix    "${OUTDIR}/phase30" \
        && log "  Figura de forma: output/phase30_shape_comparison.png" \
        || warn "  Error en compare_pk_shape.py"
    else
        warn "  P(k) inicial no encontrado — omitiendo comparación de forma"
    fi
fi

# ── Paso 5: Validar crecimiento vs D1(a) ─────────────────────────────────────

log "=== Paso 5: Validar crecimiento vs D1(a) ==="

if [[ -n "${PYTHON}" ]]; then
    SNAPS_1LPT=$(ls "${OUTDIR}/lcdm_N32_a002_1lpt_pm"/snap_*.json 2>/dev/null | sort || true)
    SNAPS_2LPT=$(ls "${OUTDIR}/lcdm_N32_a002_2lpt_pm"/snap_*.json 2>/dev/null | sort || true)

    if [[ -n "${SNAPS_1LPT}" ]]; then
        ARGS_2LPT_G=""
        [[ -n "${SNAPS_2LPT}" ]] && ARGS_2LPT_G="--snapshots-2lpt ${SNAPS_2LPT}"

        "${PYTHON}" "${SCRIPTS}/plot_growth_vs_d1.py" \
            --snapshots     ${SNAPS_1LPT} \
            --a-init        "${A_INIT}" \
            --omega-m       0.315 \
            --omega-l       0.685 \
            ${ARGS_2LPT_G} \
            --out-prefix    "${OUTDIR}/phase30" \
        && log "  Figura de crecimiento: output/phase30_growth_vs_d1.png" \
        || warn "  Error en plot_growth_vs_d1.py"
    else
        warn "  Snapshots no encontrados — omitiendo comparación de crecimiento"
    fi
fi

# ── Paso 6: Comparar 1LPT vs 2LPT vs referencia ──────────────────────────────

log "=== Paso 6: 1LPT vs 2LPT vs referencia ==="

if [[ -n "${PYTHON}" && -f "${OUTDIR}/reference_pk.json" ]]; then
    PK_1LPT="${OUTDIR}/pk_init_1lpt_pm.json"
    PK_2LPT="${OUTDIR}/pk_init_2lpt_pm.json"

    if [[ -f "${PK_1LPT}" && -f "${PK_2LPT}" ]]; then
        SNAPS_1LPT=$(ls "${OUTDIR}/lcdm_N32_a002_1lpt_pm"/snap_*.json 2>/dev/null | sort || true)
        SNAPS_2LPT=$(ls "${OUTDIR}/lcdm_N32_a002_2lpt_pm"/snap_*.json 2>/dev/null | sort || true)

        ARGS_SNAPS=""
        [[ -n "${SNAPS_1LPT}" ]] && ARGS_SNAPS="--snaps-1lpt ${SNAPS_1LPT}"
        [[ -n "${SNAPS_2LPT}" ]] && ARGS_SNAPS="${ARGS_SNAPS} --snaps-2lpt ${SNAPS_2LPT}"

        "${PYTHON}" "${SCRIPTS}/plot_1lpt_vs_2lpt_reference.py" \
            --pk-1lpt    "${PK_1LPT}" \
            --pk-2lpt    "${PK_2LPT}" \
            --pk-ref     "${OUTDIR}/reference_pk.json" \
            --box        "${BOX_MPC_H}" \
            --h          "${H_DIMLESS}" \
            --a-init     "${A_INIT}" \
            ${ARGS_SNAPS} \
            --out-prefix "${OUTDIR}/phase30" \
        && log "  Figura: output/phase30_1lpt_vs_2lpt_reference.png" \
        || warn "  Error en plot_1lpt_vs_2lpt_reference.py"
    fi
fi

# ── Paso 7: Ejecutar tests automáticos de Phase 30 ───────────────────────────

log "=== Paso 7: Tests automáticos ==="
(cd "${REPO_ROOT}" && cargo test --test phase30_linear_reference -- --nocapture 2>&1 \
    | tee "${OUTDIR}/test_results.txt") \
&& log "  Todos los tests pasaron" \
|| warn "  Algunos tests fallaron — ver output/test_results.txt"

# ── Resumen ───────────────────────────────────────────────────────────────────

log "=== Resumen de la Fase 30 ==="
echo ""
echo "  Archivos generados en output/:"
ls -la "${OUTDIR}/" 2>/dev/null | grep -E "\.(json|png|txt)$" || echo "  (ninguno)"
echo ""
echo "  Figuras:"
for fig in "${OUTDIR}"/*.png; do
    [[ -f "${fig}" ]] && echo "    - $(basename ${fig})"
done
echo ""
echo "  Tests: ver output/test_results.txt"
echo ""
echo "  NOTA IMPORTANTE — Normalización absoluta:"
echo "  El P(k) medido de partículas NO coincide en amplitud con P_EH(k)"
echo "  (ratio R ~ 2e-4, constante en todos los k-bins)."
echo "  Esto es esperado: ver docs/reports/2026-04-phase30-linear-reference-validation.md"
echo "  La validación de FORMA y CRECIMIENTO sí es válida."
echo ""

log "Fase 30 completada."
