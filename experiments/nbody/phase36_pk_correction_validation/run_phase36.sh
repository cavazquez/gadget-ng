#!/usr/bin/env bash
# ── Phase 36 — Validación práctica de pk_correction ─────────────────────────
#
# Orquestador híbrido:
#   1. tests Rust in-process (matriz completa, release)
#   2. pase CLI evidencial (stepping + analyse + apply)
#   3. generación de figuras
#   4. copia de figuras a docs/reports/figures/phase36/
#
# Uso:
#   ./run_phase36.sh              # todo el pipeline
#   SKIP_TESTS=1 ./run_phase36.sh # saltar tests Rust (sólo regenerar figuras)
#   SKIP_CLI=1   ./run_phase36.sh # saltar pase CLI

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/output"
FIG_DIR="${SCRIPT_DIR}/figures"
DOCS_FIG="${REPO_ROOT}/docs/reports/figures/phase36"
CONFIG="${SCRIPT_DIR}/configs/lcdm_N32_2lpt_pm_phase36.toml"
BINARY="${REPO_ROOT}/target/release/gadget-ng"
PYTHON="${PYTHON:-python3}"

log() { printf "\033[1;34m[phase36]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[phase36 ! ]\033[0m %s\n" "$*"; }

mkdir -p "${OUT_DIR}" "${FIG_DIR}" "${DOCS_FIG}"

# ── Paso 1 — Tests Rust in-process ──────────────────────────────────────────

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
    log "Ejecutando tests Rust (matriz completa, release)…"
    (cd "${REPO_ROOT}" && cargo test -p gadget-ng-physics --release \
        --test phase36_pk_correction_validation -- --test-threads=1)
    log "Tests Rust OK."
else
    warn "SKIP_TESTS=1 — omitiendo tests Rust."
fi

MATRIX_JSON="${REPO_ROOT}/target/phase36/per_snapshot_metrics.json"
if [[ ! -f "${MATRIX_JSON}" ]]; then
    warn "No se encontró ${MATRIX_JSON}. Corrí los tests Rust primero."
    exit 1
fi

# ── Paso 2 — Pase CLI evidencial (N=32 2LPT seed 42) ───────────────────────

CLI_JSON="${OUT_DIR}/cli_evidence.json"
if [[ "${SKIP_CLI:-0}" != "1" ]]; then
    log "Compilando binario gadget-ng (release)…"
    (cd "${REPO_ROOT}" && cargo build --release --bin gadget-ng)

    CLI_SNAP_DIR="${OUT_DIR}/cli_snapshot"
    ANALYSE_DIR="${OUT_DIR}/cli_analysis"
    rm -rf "${CLI_SNAP_DIR}" "${ANALYSE_DIR}"
    mkdir -p "${CLI_SNAP_DIR}" "${ANALYSE_DIR}"

    # Generamos el snapshot IC real vía CLI. Evitamos `stepping` porque la
    # convención de ICs con σ₈=0.8 aplicada en `a_init` (ver reporte §2 y
    # `lcdm_N32_a005_2lpt_pm.toml` de Phase 30) produce desplazamientos
    # ~50× mayores que el régimen lineal a z≈49, lo cual hace explotar FoF
    # y P(k) tras evolucionar. La validación cuantitativa del crecimiento
    # cae fuera del alcance de Phase 36 (está centrada en amplitud absoluta).
    log "Generando snapshot IC con gadget-ng snapshot…"
    "${BINARY}" snapshot \
        --config "${CONFIG}" \
        --out "${CLI_SNAP_DIR}"

    log "Corriendo gadget-ng analyse…"
    # linking-length grande → FoF con pocas celdas (evita una OOB
    # latente en `fof.rs` con particles ZA de alta amplitud). No afecta
    # el P(k) que es lo que validamos.
    "${BINARY}" analyse \
        --snapshot "${CLI_SNAP_DIR}" \
        --out "${ANALYSE_DIR}" \
        --pk-mesh 32 \
        --linking-length 1.0

    log "Aplicando pk_correction a power_spectrum.jsonl…"
    "${PYTHON}" "${SCRIPT_DIR}/scripts/apply_phase36_correction.py" \
        --pk-jsonl "${ANALYSE_DIR}/power_spectrum.jsonl" \
        --n 32 \
        --box-internal 1.0 \
        --box-mpc-h 100.0 \
        --a-snapshot 0.02 \
        --a-init 0.02 \
        --out "${CLI_JSON}"
else
    warn "SKIP_CLI=1 — omitiendo pase CLI."
fi

# ── Paso 3 — Figuras ────────────────────────────────────────────────────────

log "Generando figuras…"
if [[ -f "${CLI_JSON}" ]]; then
    "${PYTHON}" "${SCRIPT_DIR}/scripts/plot_phase36.py" \
        --matrix-json "${MATRIX_JSON}" \
        --cli-json "${CLI_JSON}" \
        --out-dir "${FIG_DIR}"
else
    "${PYTHON}" "${SCRIPT_DIR}/scripts/plot_phase36.py" \
        --matrix-json "${MATRIX_JSON}" \
        --out-dir "${FIG_DIR}"
fi

# ── Paso 4 — Copia a docs ───────────────────────────────────────────────────

log "Copiando figuras a ${DOCS_FIG}…"
cp -f "${FIG_DIR}"/*.png "${DOCS_FIG}/"

log "Phase 36 completo. Figuras en ${FIG_DIR}/ y ${DOCS_FIG}/."
