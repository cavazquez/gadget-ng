#!/usr/bin/env bash
# run_phase31.sh — Fase 31: Validación estadística por ensemble a mayor resolución.
#
# Orquesta todo el pipeline:
#   1. Build gadget-ng (release)
#   2. Genera referencia EH (reutiliza script de Phase 30)
#   3. Para cada configuración base × seed: genera TOML, ejecuta gadget-ng,
#      extrae P(k) del snapshot inicial
#   4. Calcula estadísticas del ensemble (compute_ensemble_stats.py)
#   5. Genera figuras (plot_ensemble_pk.py, plot_lpt_pm_comparisons.py)
#
# Uso:
#   bash run_phase31.sh [--skip-build] [--skip-n64] [--only-test]
#
# Salida: experiments/nbody/phase31_ensemble_higher_res/output/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIGS_DIR="${SCRIPT_DIR}/configs"
SCRIPTS_DIR="${SCRIPT_DIR}/scripts"
OUT_DIR="${SCRIPT_DIR}/output"
FIGS_DIR="${SCRIPT_DIR}/figures"

# ── Configuración del ensemble ────────────────────────────────────────────────

SEEDS=(42 137 271 314)               # 4 seeds principales
SEEDS_N64=(42 137)                   # solo 2 seeds para N=64³ (más costoso)

SKIP_BUILD=false
SKIP_N64=false
ONLY_TEST=false

for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=true ;;
        --skip-n64)   SKIP_N64=true ;;
        --only-test)  ONLY_TEST=true ;;
    esac
done

# ── Colores para logs ─────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${BLUE}[phase31]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# ── Crear directorios de salida ───────────────────────────────────────────────

mkdir -p "${OUT_DIR}" "${FIGS_DIR}"

# ── Paso 0: Build (opcional) ──────────────────────────────────────────────────

GADGET_BIN="${REPO_ROOT}/target/release/gadget-ng"

if [[ "${SKIP_BUILD}" == "false" ]]; then
    log "Compilando gadget-ng (release)..."
    cd "${REPO_ROOT}"
    cargo build --release -p gadget-ng 2>&1 | tail -5
    ok "Build completado: ${GADGET_BIN}"
else
    warn "Saltando build (--skip-build)"
fi

if [[ ! -f "${GADGET_BIN}" ]]; then
    echo "ERROR: gadget-ng no encontrado en ${GADGET_BIN}"
    echo "Ejecutar sin --skip-build o compilar manualmente."
    exit 1
fi

# ── Modo solo-test (ejecuta los tests de Rust) ────────────────────────────────

if [[ "${ONLY_TEST}" == "true" ]]; then
    log "Ejecutando tests de Phase 31..."
    cd "${REPO_ROOT}"
    cargo test -p gadget-ng-physics --test phase31_ensemble -- --nocapture
    ok "Tests completados"
    exit 0
fi

# ── Paso 1: Referencia EH ─────────────────────────────────────────────────────

REF_PK_SCRIPT="${REPO_ROOT}/experiments/nbody/phase30_linear_reference/scripts/generate_reference_pk.py"
REF_JSON="${OUT_DIR}/reference_pk_eh.json"

if [[ -f "${REF_PK_SCRIPT}" ]]; then
    log "Generando referencia EH..."
    python3 "${REF_PK_SCRIPT}" \
        --box-mpc-h 100.0 \
        --n-k 64 \
        --output "${REF_JSON}" \
        --sigma8 0.8 \
        --n-s 0.965 2>/dev/null || true
    [[ -f "${REF_JSON}" ]] && ok "Referencia EH: ${REF_JSON}" || warn "Referencia EH no generada (script no disponible)"
else
    warn "Script de referencia EH no encontrado: ${REF_PK_SCRIPT}"
    warn "Continuando sin referencia externa (se usará EH interno)"
fi

# ── Función: ejecutar una corrida y extraer P(k) inicial ─────────────────────

run_single() {
    local base_config="$1"
    local seed="$2"
    local out_prefix="$3"

    local tmp_config="${OUT_DIR}/tmp_$(basename ${base_config%.toml})_seed${seed}.toml"
    local run_dir="${OUT_DIR}/run_$(basename ${base_config%.toml})_seed${seed}"

    # Generar config con seed específico (sustituye SEED=42 por el seed actual)
    sed "s/seed *= *42/seed = ${seed}/g; s/seed *= *42,/seed = ${seed},/g" \
        "${base_config}" > "${tmp_config}"

    mkdir -p "${run_dir}"

    log "  Ejecutando: $(basename ${base_config%.toml}) seed=${seed}..."

    # Ejecutar gadget-ng
    "${GADGET_BIN}" "${tmp_config}" --output-dir "${run_dir}" \
        2>"${run_dir}/stderr.log" || {
        warn "  gadget-ng falló para $(basename ${base_config%.toml}) seed=${seed}"
        warn "  Ver: ${run_dir}/stderr.log"
        return 1
    }

    # Buscar snapshot inicial (paso 0 o el más temprano)
    local snap_init=$(ls "${run_dir}"/snapshot_0*.json 2>/dev/null | head -1 || true)
    if [[ -z "${snap_init}" ]]; then
        # Intentar formato alternativo
        snap_init=$(ls "${run_dir}"/*.json 2>/dev/null | head -1 || true)
    fi

    if [[ -n "${snap_init}" ]]; then
        cp "${snap_init}" "${OUT_DIR}/${out_prefix}_seed${seed}.json"
        ok "  → P(k) inicial: ${OUT_DIR}/${out_prefix}_seed${seed}.json"
    else
        warn "  No se encontró snapshot inicial en ${run_dir}"
    fi

    # Limpiar config temporal
    rm -f "${tmp_config}"
}

# ── Paso 2: Ejecutar todas las corridas N=32³ ─────────────────────────────────

log "=== Ejecutando corridas N=32³ (4 seeds × 4 configs) ==="

CONFIGS_N32=(
    "base_N32_a002_1lpt_pm"
    "base_N32_a002_2lpt_pm"
    "base_N32_a002_2lpt_treepm"
    "base_N32_a005_2lpt_pm"
)

for cfg_name in "${CONFIGS_N32[@]}"; do
    cfg_path="${CONFIGS_DIR}/${cfg_name}.toml"
    if [[ ! -f "${cfg_path}" ]]; then
        warn "Config no encontrado: ${cfg_path}"
        continue
    fi
    log "Procesando: ${cfg_name}"
    for seed in "${SEEDS[@]}"; do
        run_single "${cfg_path}" "${seed}" "${cfg_name}" || true
    done
done

# ── Paso 3: Ejecutar corridas N=64³ (opcional) ───────────────────────────────

if [[ "${SKIP_N64}" == "false" ]]; then
    log "=== Ejecutando corridas N=64³ (2 seeds) ==="
    cfg_path="${CONFIGS_DIR}/base_N64_a002_2lpt_pm.toml"
    if [[ -f "${cfg_path}" ]]; then
        for seed in "${SEEDS_N64[@]}"; do
            run_single "${cfg_path}" "${seed}" "base_N64_a002_2lpt_pm" || true
        done
    else
        warn "Config N=64³ no encontrado, saltando"
    fi
else
    warn "Saltando corridas N=64³ (--skip-n64)"
fi

# ── Paso 4: Calcular estadísticas del ensemble ────────────────────────────────

log "=== Calculando estadísticas del ensemble ==="

compute_stats() {
    local prefix="$1"
    local label="$2"
    local box_mpc_h="${3:-100.0}"

    # Recopilar archivos de P(k) para este prefijo
    local pk_files=()
    for seed in "${SEEDS[@]}"; do
        local f="${OUT_DIR}/${prefix}_seed${seed}.json"
        [[ -f "${f}" ]] && pk_files+=("${f}")
    done

    if [[ ${#pk_files[@]} -eq 0 ]]; then
        warn "No hay archivos de P(k) para ${prefix}"
        return
    fi

    log "  ${label}: ${#pk_files[@]} seeds"
    python3 "${SCRIPTS_DIR}/compute_ensemble_stats.py" \
        --pk-files "${pk_files[@]}" \
        --label "${label}" \
        --box-mpc-h "${box_mpc_h}" \
        --sigma8 0.8 --n-s 0.965 \
        --output "${OUT_DIR}/stats_${label}.json"
}

compute_stats "base_N32_a002_1lpt_pm"    "N32_a002_1lpt_pm"
compute_stats "base_N32_a002_2lpt_pm"    "N32_a002_2lpt_pm"
compute_stats "base_N32_a002_2lpt_treepm" "N32_a002_2lpt_treepm"
compute_stats "base_N32_a005_2lpt_pm"    "N32_a005_2lpt_pm"

if [[ "${SKIP_N64}" == "false" ]]; then
    # Stats N64 solo con 2 seeds
    pk_files_n64=()
    for seed in "${SEEDS_N64[@]}"; do
        f="${OUT_DIR}/base_N64_a002_2lpt_pm_seed${seed}.json"
        [[ -f "${f}" ]] && pk_files_n64+=("${f}")
    done
    if [[ ${#pk_files_n64[@]} -gt 0 ]]; then
        python3 "${SCRIPTS_DIR}/compute_ensemble_stats.py" \
            --pk-files "${pk_files_n64[@]}" \
            --label "N64_a002_2lpt_pm" \
            --box-mpc-h 100.0 \
            --sigma8 0.8 --n-s 0.965 \
            --output "${OUT_DIR}/stats_N64_a002_2lpt_pm.json"
    fi
fi

# ── Paso 5: Generar figuras ───────────────────────────────────────────────────

log "=== Generando figuras ==="

# Figura 1+2+3: P_mean(k) y R(k) para N=32³
if [[ -f "${OUT_DIR}/stats_N32_a002_2lpt_pm.json" ]]; then
    python3 "${SCRIPTS_DIR}/plot_ensemble_pk.py" \
        --stats-n32 "${OUT_DIR}/stats_N32_a002_2lpt_pm.json" \
        $([ -f "${OUT_DIR}/stats_N64_a002_2lpt_pm.json" ] && \
          echo "--stats-n64 ${OUT_DIR}/stats_N64_a002_2lpt_pm.json") \
        --output-dir "${FIGS_DIR}" || warn "plot_ensemble_pk.py falló (matplotlib requerido)"
fi

# Figura: comparaciones 1LPT vs 2LPT y PM vs TreePM
if [[ -f "${OUT_DIR}/stats_N32_a002_1lpt_pm.json" ]] && \
   [[ -f "${OUT_DIR}/stats_N32_a002_2lpt_pm.json" ]]; then
    python3 "${SCRIPTS_DIR}/plot_lpt_pm_comparisons.py" \
        --stats-1lpt "${OUT_DIR}/stats_N32_a002_1lpt_pm.json" \
        --stats-2lpt "${OUT_DIR}/stats_N32_a002_2lpt_pm.json" \
        $([ -f "${OUT_DIR}/stats_N32_a002_2lpt_treepm.json" ] && \
          echo "--stats-pm ${OUT_DIR}/stats_N32_a002_2lpt_pm.json \
                --stats-treepm ${OUT_DIR}/stats_N32_a002_2lpt_treepm.json") \
        --output-dir "${FIGS_DIR}" || warn "plot_lpt_pm_comparisons.py falló"
fi

# ── Paso 6: Tests de Rust (resumen) ──────────────────────────────────────────

log "=== Ejecutando tests de Phase 31 (Rust) ==="
cd "${REPO_ROOT}"
cargo test -p gadget-ng-physics --test phase31_ensemble 2>&1 | tail -5

# ── Resumen final ─────────────────────────────────────────────────────────────

echo ""
ok "=== Phase 31 completada ==="
echo "  Output: ${OUT_DIR}"
echo "  Figuras: ${FIGS_DIR}"
echo ""

log "Estadísticas generadas:"
for f in "${OUT_DIR}"/stats_*.json; do
    [[ -f "${f}" ]] && echo "  $(basename ${f})"
done

log "Figuras generadas:"
for f in "${FIGS_DIR}"/*.png; do
    [[ -f "${f}" ]] && echo "  $(basename ${f})"
done

echo ""
echo "Siguiente paso: revisar el reporte en docs/reports/2026-04-phase31-*.md"
