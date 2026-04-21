#!/usr/bin/env bash
# ── Phase 39 — Convergencia temporal del integrador Leapfrog KDK ────────────
#
# Orquestador híbrido:
#   1. Tests Rust in-process (matriz 4 dt × 3 seeds × 3 a = 36 mediciones).
#   2. Pase CLI evidencial para los 4 dts, seed 42, hasta a ≈ 0.10.
#   3. Generación de 4 figuras + CSV desde el JSON dumpeado.
#   4. Copia de figuras a docs/reports/figures/phase39/.
#
# Uso:
#   ./run_phase39.sh               # pipeline completo
#   SKIP_TESTS=1 ./run_phase39.sh  # saltar tests Rust (sólo regenerar figuras)
#   SKIP_CLI=1   ./run_phase39.sh  # saltar pase CLI
#   SKIP_FIGS=1  ./run_phase39.sh  # saltar figuras

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/output"
FIG_DIR="${SCRIPT_DIR}/figures"
DOCS_FIG="${REPO_ROOT}/docs/reports/figures/phase39"
BINARY="${REPO_ROOT}/target/release/gadget-ng"
PYTHON="${PYTHON:-python3}"

DTS=("4e4" "2e4" "1e4" "5e5")
DT_VALUES=(4.0e-4 2.0e-4 1.0e-4 5.0e-5)

log()  { printf "\033[1;34m[phase39]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[phase39 ! ]\033[0m %s\n" "$*"; }

mkdir -p "${OUT_DIR}" "${FIG_DIR}" "${DOCS_FIG}"

# ── Paso 1 — Tests Rust in-process (matriz 36 mediciones) ───────────────────

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
    log "Ejecutando tests Rust Phase 39 (4 dt × 3 seeds × 3 a, release)…"
    (cd "${REPO_ROOT}" && cargo test -p gadget-ng-physics --release \
        --test phase39_dt_convergence -- --test-threads=1)
    log "Tests Rust OK."
else
    warn "SKIP_TESTS=1 — omitiendo tests Rust."
fi

MATRIX_JSON="${REPO_ROOT}/target/phase39/per_cfg.json"
if [[ ! -f "${MATRIX_JSON}" ]]; then
    warn "No se encontró ${MATRIX_JSON}. Corré los tests Rust primero."
    exit 1
fi

# ── Paso 2 — Pase CLI evidencial (seed 42, los 4 dts) ───────────────────────

if [[ "${SKIP_CLI:-0}" != "1" ]]; then
    log "Compilando binario gadget-ng (release)…"
    (cd "${REPO_ROOT}" && cargo build --release --bin gadget-ng)

    for i in 0 1 2 3; do
        tag="${DTS[$i]}"
        dt="${DT_VALUES[$i]}"
        CFG="${SCRIPT_DIR}/configs/lcdm_N32_2lpt_pm_dt_${tag}.toml"
        RUN_DIR="${OUT_DIR}/dt_${tag}/cli_run"
        ANA_DIR="${OUT_DIR}/dt_${tag}/cli_analysis"
        EVID="${OUT_DIR}/dt_${tag}/cli_evidence.json"

        rm -rf "${RUN_DIR}" "${ANA_DIR}"
        mkdir -p "${RUN_DIR}" "${ANA_DIR}" "$(dirname "${EVID}")"

        log "[dt=${dt}] gadget-ng stepping (con snapshot final)…"
        "${BINARY}" stepping --config "${CFG}" --out "${RUN_DIR}" --snapshot \
            > "${OUT_DIR}/dt_${tag}/stepping.log" 2>&1 \
            || { warn "stepping falló (ver ${OUT_DIR}/dt_${tag}/stepping.log)"; exit 1; }

        SNAP_FINAL="${RUN_DIR}/snapshot_final"
        if [[ ! -d "${SNAP_FINAL}" ]]; then
            warn "No encontré ${SNAP_FINAL}; chequeá stepping.log"
            exit 1
        fi

        log "[dt=${dt}] gadget-ng analyse sobre ${SNAP_FINAL}…"
        "${BINARY}" analyse \
            --snapshot "${SNAP_FINAL}" \
            --out "${ANA_DIR}" \
            --pk-mesh 32 \
            --linking-length 1.0 \
            > "${OUT_DIR}/dt_${tag}/analyse.log" 2>&1 \
            || { warn "analyse falló"; exit 1; }

        # El snapshot CLI evoluciona num_steps · dt desde a_init; usamos esa
        # aproximación de primer orden (coincide con `advance_a` dentro de
        # la tolerancia del sweep).
        NSTEPS="$(awk -F= '/^num_steps/ { gsub(" ",""); print $2 }' "${CFG}")"
        A_FINAL="$(python3 -c "print(0.02 + ${NSTEPS} * ${dt})")"

        log "[dt=${dt}] apply_phase39_correction.py (a≈${A_FINAL})…"
        "${PYTHON}" "${SCRIPT_DIR}/scripts/apply_phase39_correction.py" \
            --pk-jsonl "${ANA_DIR}/power_spectrum.jsonl" \
            --n 32 \
            --box-internal 1.0 \
            --box-mpc-h 100.0 \
            --dt "${dt}" \
            --a-snapshot "${A_FINAL}" \
            --a-init 0.02 \
            --out "${EVID}"
    done
else
    warn "SKIP_CLI=1 — omitiendo pase CLI."
fi

# ── Paso 3 — Figuras + CSV ───────────────────────────────────────────────────

if [[ "${SKIP_FIGS:-0}" != "1" ]]; then
    log "Generando figuras + CSV desde ${MATRIX_JSON}…"
    "${PYTHON}" "${SCRIPT_DIR}/scripts/plot_dt_sweep.py" \
        --per-cfg "${MATRIX_JSON}" \
        --out-dir "${FIG_DIR}" \
        --csv "${OUT_DIR}/dt_vs_error.csv"
else
    warn "SKIP_FIGS=1 — omitiendo figuras."
fi

# ── Paso 4 — Copia a docs ───────────────────────────────────────────────────

log "Copiando figuras a ${DOCS_FIG}…"
cp -f "${FIG_DIR}"/*.png "${DOCS_FIG}/" 2>/dev/null || warn "Sin figuras nuevas."

log "Phase 39 completo."
log "  matriz JSON : ${MATRIX_JSON}"
log "  CSV         : ${OUT_DIR}/dt_vs_error.csv"
log "  figures     : ${FIG_DIR}/ y ${DOCS_FIG}/"
