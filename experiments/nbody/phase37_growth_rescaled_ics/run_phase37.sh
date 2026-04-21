#!/usr/bin/env bash
# ── Phase 37 — Reescalado físico opcional de ICs por D(a_init)/D(1) ────────
#
# Orquestador híbrido:
#   1. Tests Rust in-process (matriz 90 snapshots legacy vs rescaled).
#   2. Pase CLI evidencial para los dos modos (legacy y rescaled).
#   3. Generación de figuras desde el JSON dumpeado por el test Rust.
#   4. Copia de figuras a docs/reports/figures/phase37/.
#
# Uso:
#   ./run_phase37.sh               # pipeline completo
#   SKIP_TESTS=1 ./run_phase37.sh  # saltar tests Rust (sólo regenerar figuras)
#   SKIP_CLI=1   ./run_phase37.sh  # saltar pase CLI dual

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/output"
FIG_DIR="${SCRIPT_DIR}/figures"
DOCS_FIG="${REPO_ROOT}/docs/reports/figures/phase37"
CFG_LEGACY="${SCRIPT_DIR}/configs/lcdm_N32_2lpt_pm_legacy.toml"
CFG_RESCALED="${SCRIPT_DIR}/configs/lcdm_N32_2lpt_pm_rescaled.toml"
BINARY="${REPO_ROOT}/target/release/gadget-ng"
PYTHON="${PYTHON:-python3}"

log()  { printf "\033[1;34m[phase37]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[phase37 ! ]\033[0m %s\n" "$*"; }

mkdir -p "${OUT_DIR}" "${FIG_DIR}" "${DOCS_FIG}"

# ── Paso 1 — Tests Rust in-process (matriz 90 snapshots) ───────────────────

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
    log "Ejecutando tests Rust Phase 37 (matriz 5 configs × 3 seeds × 3 a × 2 modes, release)…"
    (cd "${REPO_ROOT}" && cargo test -p gadget-ng-physics --release \
        --test phase37_growth_rescaled_ics -- --test-threads=1)
    log "Tests Rust OK."
else
    warn "SKIP_TESTS=1 — omitiendo tests Rust."
fi

MATRIX_JSON="${REPO_ROOT}/target/phase37/per_snapshot_metrics.json"
if [[ ! -f "${MATRIX_JSON}" ]]; then
    warn "No se encontró ${MATRIX_JSON}. Corré los tests Rust primero."
    exit 1
fi

# ── Paso 2 — Pase CLI dual (legacy + rescaled) ─────────────────────────────

if [[ "${SKIP_CLI:-0}" != "1" ]]; then
    log "Compilando binario gadget-ng (release)…"
    (cd "${REPO_ROOT}" && cargo build --release --bin gadget-ng)

    for mode in legacy rescaled; do
        if [[ "${mode}" == "legacy" ]]; then
            CFG="${CFG_LEGACY}"
        else
            CFG="${CFG_RESCALED}"
        fi
        SNAP_DIR="${OUT_DIR}/${mode}/cli_snapshot"
        ANA_DIR="${OUT_DIR}/${mode}/cli_analysis"
        EVID="${OUT_DIR}/${mode}/cli_evidence.json"

        rm -rf "${SNAP_DIR}" "${ANA_DIR}"
        mkdir -p "${SNAP_DIR}" "${ANA_DIR}" "$(dirname "${EVID}")"

        log "[${mode}] Generando snapshot IC con gadget-ng snapshot…"
        "${BINARY}" snapshot --config "${CFG}" --out "${SNAP_DIR}"

        log "[${mode}] Corriendo gadget-ng analyse…"
        "${BINARY}" analyse \
            --snapshot "${SNAP_DIR}" \
            --out "${ANA_DIR}" \
            --pk-mesh 32 \
            --linking-length 1.0

        log "[${mode}] Aplicando pk_correction + referencia ${mode}…"
        "${PYTHON}" "${SCRIPT_DIR}/scripts/apply_phase37_correction.py" \
            --pk-jsonl "${ANA_DIR}/power_spectrum.jsonl" \
            --n 32 \
            --box-internal 1.0 \
            --box-mpc-h 100.0 \
            --a-snapshot 0.02 \
            --a-init 0.02 \
            --mode "${mode}" \
            --out "${EVID}"
    done
else
    warn "SKIP_CLI=1 — omitiendo pase CLI."
fi

# ── Paso 3 — Figuras ────────────────────────────────────────────────────────

log "Generando figuras desde ${MATRIX_JSON}…"
"${PYTHON}" "${SCRIPT_DIR}/scripts/plot_phase37.py" \
    --metrics "${MATRIX_JSON}" \
    --out-dir "${FIG_DIR}"

# ── Paso 4 — Copia a docs ───────────────────────────────────────────────────

log "Copiando figuras a ${DOCS_FIG}…"
cp -f "${FIG_DIR}"/*.png "${DOCS_FIG}/" 2>/dev/null || warn "Sin figuras nuevas."

log "Phase 37 completo. Figuras en ${FIG_DIR}/ y ${DOCS_FIG}/."
