#!/usr/bin/env bash
# ── Phase 38 — External validation vs CLASS ────────────────────────────────
#
# Orquestador híbrido:
#   1. Chequeo de referencia CLASS (.dat + SHA256 esperado).
#   2. Tests Rust in-process (matriz 2 N × 3 seeds × 2 modos = 12 mediciones).
#   3. Pase CLI evidencial para los dos modos (legacy vs z=0, rescaled vs z=49).
#   4. Generación de figuras desde el JSON dumpeado por el test Rust.
#   5. Copia de figuras a docs/reports/figures/phase38/.
#
# Uso:
#   ./run_phase38.sh               # pipeline completo
#   SKIP_TESTS=1 ./run_phase38.sh  # saltar tests Rust (sólo regenerar figuras)
#   SKIP_CLI=1   ./run_phase38.sh  # saltar pase CLI dual

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/output"
FIG_DIR="${SCRIPT_DIR}/figures"
DOCS_FIG="${REPO_ROOT}/docs/reports/figures/phase38"
REF_DIR="${SCRIPT_DIR}/reference"
CFG_LEGACY="${SCRIPT_DIR}/configs/lcdm_N32_2lpt_pm_legacy.toml"
CFG_RESCALED="${SCRIPT_DIR}/configs/lcdm_N32_2lpt_pm_rescaled.toml"
CLASS_Z0="${REF_DIR}/pk_class_z0.dat"
CLASS_Z49="${REF_DIR}/pk_class_z49.dat"
BINARY="${REPO_ROOT}/target/release/gadget-ng"
PYTHON="${PYTHON:-python3}"

log()  { printf "\033[1;34m[phase38]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[phase38 ! ]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[phase38 X ]\033[0m %s\n" "$*" >&2; exit 1; }

mkdir -p "${OUT_DIR}" "${FIG_DIR}" "${DOCS_FIG}"

# ── Paso 0 — Chequeo de referencia CLASS ────────────────────────────────────

for dat in "${CLASS_Z0}" "${CLASS_Z49}"; do
    if [[ ! -f "${dat}" ]]; then
        die "Referencia CLASS faltante: ${dat}
  → Corré ${REF_DIR}/generate_reference.sh para regenerar las tablas."
    fi
done
log "CLASS reference tables:"
( cd "${REF_DIR}" && sha256sum "$(basename "${CLASS_Z0}")" "$(basename "${CLASS_Z49}")" )

# ── Paso 1 — Tests Rust in-process (matriz 12 mediciones) ──────────────────

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
    log "Ejecutando tests Rust Phase 38 (2 N × 3 seeds × 2 modos, release)…"
    (cd "${REPO_ROOT}" && cargo test -p gadget-ng-physics --release \
        --test phase38_class_validation -- --test-threads=1)
    log "Tests Rust OK."
else
    warn "SKIP_TESTS=1 — omitiendo tests Rust."
fi

MATRIX_JSON="${REPO_ROOT}/target/phase38/per_measurement.json"
if [[ ! -f "${MATRIX_JSON}" ]]; then
    die "No se encontró ${MATRIX_JSON}. Corré los tests Rust primero."
fi

# ── Paso 2 — Pase CLI dual (legacy vs z=0, rescaled vs z=49) ──────────────

if [[ "${SKIP_CLI:-0}" != "1" ]]; then
    log "Compilando binario gadget-ng (release)…"
    (cd "${REPO_ROOT}" && cargo build --release --bin gadget-ng)

    for mode in legacy rescaled; do
        if [[ "${mode}" == "legacy" ]]; then
            CFG="${CFG_LEGACY}"
            CLASS_REF="${CLASS_Z0}"
        else
            CFG="${CFG_RESCALED}"
            CLASS_REF="${CLASS_Z49}"
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

        log "[${mode}] Aplicando pk_correction + comparando vs $(basename "${CLASS_REF}")…"
        # Nota: NO se pasa --apply-unit-factor. El modelo `R(N)` de Phase 35 se
        # calibró contra P_cont en (Mpc/h)^3 con box_internal=1, absorbiendo el
        # factor de unidades implícitamente (ver Phase 36 §2). Duplicarlo mete
        # un factor 10^6 espurio.
        "${PYTHON}" "${SCRIPT_DIR}/scripts/apply_phase38_correction.py" \
            --pk-jsonl "${ANA_DIR}/power_spectrum.jsonl" \
            --class-ref "${CLASS_REF}" \
            --mode "${mode}" \
            --n 32 \
            --box-internal 1.0 \
            --box-mpc-h 100.0 \
            --out "${EVID}"
    done
else
    warn "SKIP_CLI=1 — omitiendo pase CLI."
fi

# ── Paso 3 — Figuras ────────────────────────────────────────────────────────

log "Generando figuras desde ${MATRIX_JSON}…"
"${PYTHON}" "${SCRIPT_DIR}/scripts/plot_phase38.py" \
    --metrics "${MATRIX_JSON}" \
    --out-dir "${FIG_DIR}"

# ── Paso 4 — Copia a docs ───────────────────────────────────────────────────

log "Copiando figuras a ${DOCS_FIG}…"
cp -f "${FIG_DIR}"/*.png "${DOCS_FIG}/" 2>/dev/null || warn "Sin figuras nuevas."

log "Phase 38 completo. Figuras en ${FIG_DIR}/ y ${DOCS_FIG}/."
