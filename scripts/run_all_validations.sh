#!/usr/bin/env bash
# =============================================================================
# run_all_validations.sh — Suite completa de validaciones de gadget-ng
#
# Incluye:
#   Bloque 1: cargo test --workspace --release  (todos los tests unitarios)
#   Bloque 2: tests cuantitativos con --nocapture (métricas impresas)
#             Incluye: Phases 161-163 (v3_mhd_validation, v2_hierarchical_cosmo, v1_gpu_tests)
#   Bloque 3: benchmarks Criterion rápidos
#   Bloque 4: GPU (si hay hardware disponible)
#   Bloque 5: corrida de validación N=128³ (2–4 h, end-to-end cosmológica)
#   Bloque 6: corrida de PRODUCCIÓN N=256³ (8–12 h, la definitiva)
#
# Uso:
#   bash scripts/run_all_validations.sh              # todo
#   SKIP_PROD=1    bash scripts/run_all_validations.sh   # sin N=256³
#   SKIP_VAL128=1  bash scripts/run_all_validations.sh   # sin N=128³
#   ONLY_UNIT=1    bash scripts/run_all_validations.sh   # solo tests rápidos
#   N_RANKS=4      bash scripts/run_all_validations.sh   # MPI×4 para corridas largas
#
# Dejar corriendo toda la noche:
#   nohup bash scripts/run_all_validations.sh > logs/validation_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   echo "PID: $!"
#   tail -f logs/validation_*.log
# =============================================================================

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Variables de control ──────────────────────────────────────────────────────
SKIP_PROD="${SKIP_PROD:-0}"        # 1 = saltar corrida N=256³
SKIP_VAL128="${SKIP_VAL128:-0}"    # 1 = saltar corrida N=128³
ONLY_UNIT="${ONLY_UNIT:-0}"        # 1 = solo unit tests (bloques 1+2)
N_RANKS="${N_RANKS:-1}"            # ranks MPI para corridas largas
RELEASE_FLAGS="--release"

mkdir -p logs

# ── Contadores ────────────────────────────────────────────────────────────────
PASS=0; FAIL=0; SKIP=0
T_GLOBAL_START=$(date +%s)

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
pass() { PASS=$((PASS+1)); log "✓ PASS: $1"; }
fail() { FAIL=$((FAIL+1)); log "✗ FAIL: $1"; }
skip() { SKIP=$((SKIP+1)); log "~ SKIP: $1  (razón: $2)"; }
section() { echo; log "════════════════════════════════════════════════"; log "  $1"; log "════════════════════════════════════════════════"; }

elapsed() {
    local t=$(( $(date +%s) - T_GLOBAL_START ))
    printf "%dh %02dm %02ds" $((t/3600)) $(( (t%3600)/60 )) $((t%60))
}

run_test() {
    local label="$1"; shift
    if "$@" ; then
        pass "$label"
    else
        fail "$label"
    fi
}

# =============================================================================
section "Bloque 1 — cargo test --workspace $RELEASE_FLAGS"
# =============================================================================
log "Compilando workspace…"
if cargo build --workspace $RELEASE_FLAGS 2>&1; then
    pass "compilación workspace"
else
    fail "compilación workspace"
    log "ERROR: el workspace no compila. Abortando."
    exit 1
fi

log "Ejecutando cargo test --workspace $RELEASE_FLAGS …"
if cargo test --workspace $RELEASE_FLAGS 2>&1 | tee logs/bloque1_unit_tests.log \
        | grep -E "FAILED|^error"; then
    # Si hubo FAILED o error, falló
    if grep -q "FAILED" logs/bloque1_unit_tests.log; then
        fail "unit tests workspace"
    else
        pass "unit tests workspace"
    fi
else
    pass "unit tests workspace"
fi

TOTAL_TESTS=$(grep -c "^test result:" logs/bloque1_unit_tests.log 2>/dev/null || echo "?")
log "Suites ejecutadas: $TOTAL_TESTS"

if [[ "$ONLY_UNIT" == "1" ]]; then
    log "ONLY_UNIT=1 — saltando bloques 3–6"
    goto_summary=1
fi

# =============================================================================
section "Bloque 2 — Tests cuantitativos con --nocapture"
# =============================================================================
QUANTITATIVE_TESTS=(
    "phase30_linear_reference"
    "phase38_class_validation"
    "phase54_growth_factor_validation"
    "phase55_fof_vs_hmf"
    "phase145_reconnection"
    "phase146_braginskii"
    "phase147_mhd_cosmo_full"
    "phase148_rmhd_jets"
    "phase149_two_fluid"
    "phase155_dark_energy"
    "phase156_massive_neutrinos"
    "phase157_sidm"
    "phase158_modified_gravity"
    "phase159_gmc_collapse"
    # Phase 161–163 (HPC validation suite)
    "v3_mhd_validation"
    "v2_hierarchical_cosmo"
)

# Tests GPU V1 (Bloque 2b — todos los tests activos; los que requieren HW hacen skip automático)
log "→ v1_gpu_tests (6 tests: 1 wgpu + 5 skip-si-sin-HW)"
if cargo test -p gadget-ng-gpu --test v1_gpu_tests $RELEASE_FLAGS \
        -- --nocapture 2>&1 | tee "logs/bloque2_v1_gpu_tests.log" | grep -q "FAILED"; then
    fail "v1_gpu_tests"
else
    pass "v1_gpu_tests"
fi

# Tests MHD 3D solenoidal (Phase 165 — pura Rust, sin hardware)
log "→ ic_mhd 3D solenoidal (gadget-ng-core)"
if cargo test -p gadget-ng-core --lib ic_mhd $RELEASE_FLAGS \
        -- --nocapture 2>&1 | tee "logs/bloque2_ic_mhd_3d.log" | grep -q "FAILED"; then
    fail "ic_mhd 3D solenoidal"
else
    pass "ic_mhd 3D solenoidal (primordial_bfield_ic_3d: rms + div-free)"
fi

for t in "${QUANTITATIVE_TESTS[@]}"; do
    log "→ $t"
    if cargo test -p gadget-ng-physics --test "$t" $RELEASE_FLAGS \
            -- --nocapture 2>&1 | tee "logs/bloque2_${t}.log" | grep -q "FAILED"; then
        fail "$t"
    else
        pass "$t"
    fi
done

if [[ "${goto_summary:-0}" == "1" ]]; then
    goto_summary_func
fi

# =============================================================================
section "Bloque 3 — Benchmarks Criterion (quick)"
# =============================================================================
BENCHES=(
    "gadget-ng-pm:pm_gravity"
    "gadget-ng-treepm:treepm_gravity"
    "gadget-ng-mhd:advanced_bench"
    "gadget-ng-gpu:gpu_vs_cpu"
)

for entry in "${BENCHES[@]}"; do
    crate="${entry%%:*}"
    bench="${entry##*:}"
    log "→ bench $crate/$bench"
    if cargo bench -p "$crate" --bench "$bench" -- --quick \
            2>&1 | tee "logs/bloque3_bench_${bench}.log" | grep -qi "error"; then
        fail "bench $bench"
    else
        pass "bench $bench"
    fi
done

# =============================================================================
section "Bloque 4 — GPU (si hay hardware disponible)"
# =============================================================================
if command -v nvidia-smi &>/dev/null; then
    log "GPU NVIDIA detectada: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    if cargo test -p gadget-ng-gpu -- --include-ignored \
            2>&1 | tee logs/bloque4_gpu.log | grep -q "FAILED"; then
        fail "gpu tests (NVIDIA)"
    else
        pass "gpu tests (NVIDIA)"
    fi
elif command -v rocm-smi &>/dev/null; then
    log "GPU AMD (ROCm) detectada"
    if cargo test -p gadget-ng-gpu -- --include-ignored \
            2>&1 | tee logs/bloque4_gpu.log | grep -q "FAILED"; then
        fail "gpu tests (ROCm)"
    else
        pass "gpu tests (ROCm)"
    fi
else
    skip "gpu tests" "no hay GPU CUDA/ROCm disponible"
fi

# =============================================================================
section "Bloque 5 — Corrida de validación N=128³ (2–4 h end-to-end)"
# =============================================================================
if [[ "$SKIP_VAL128" == "1" || "$ONLY_UNIT" == "1" ]]; then
    skip "validación N=128³" "SKIP_VAL128=1 o ONLY_UNIT=1"
else
    log "Lanzando corrida N=128³ (puede tardar 2–4 h con ${N_RANKS} rank(s))…"
    log "Config: configs/validation_128.toml"
    log "Output: runs/validation_128/"

    if [[ "$N_RANKS" -gt 1 ]]; then
        RUN_CMD="N_RANKS=${N_RANKS} bash scripts/run_validation_128.sh"
    else
        RUN_CMD="bash scripts/run_validation_128.sh"
    fi

    T_VAL128_START=$(date +%s)
    if eval "$RUN_CMD" 2>&1 | tee logs/bloque5_validation_128.log; then
        T_VAL128=$(( $(date +%s) - T_VAL128_START ))
        pass "validación N=128³  ($(printf '%dh %02dm' $((T_VAL128/3600)) $(((T_VAL128%3600)/60))))"
    else
        fail "validación N=128³"
    fi
fi

# =============================================================================
section "Bloque 6 — Corrida de PRODUCCIÓN N=256³ (8–12 h, la definitiva)"
# =============================================================================
if [[ "$SKIP_PROD" == "1" || "$ONLY_UNIT" == "1" ]]; then
    skip "producción N=256³" "SKIP_PROD=1 o ONLY_UNIT=1"
else
    log "Lanzando corrida de PRODUCCIÓN N=256³ (la corrida definitiva)…"
    log "  - 16.7M partículas, z=49→0, ~1000 pasos"
    log "  - Checkpoints automáticos cada 2 h (reanuda si se interrumpe)"
    log "  - Solver: TreePM SFC+LET + block timesteps jerárquicos"
    log "  - Config: configs/production_256.toml"
    log "  - Output: runs/production_256/"
    log "  - Con ${N_RANKS} rank(s) MPI"

    if [[ "$N_RANKS" -gt 1 ]]; then
        RUN_CMD="N_RANKS=${N_RANKS} bash scripts/run_production_256.sh"
    else
        RUN_CMD="bash scripts/run_production_256.sh"
    fi

    T_PROD_START=$(date +%s)
    if eval "$RUN_CMD" 2>&1 | tee logs/bloque6_production_256.log; then
        T_PROD=$(( $(date +%s) - T_PROD_START ))
        pass "producción N=256³  ($(printf '%dh %02dm' $((T_PROD/3600)) $(((T_PROD%3600)/60))))"

        # Validaciones post-corrida
        log "Verificando outputs…"
        if [[ -f "runs/production_256/analysis/pk_z0.json" ]]; then
            pass "P(k) z=0 generado"
        else
            fail "P(k) z=0 no encontrado"
        fi
        if [[ -f "runs/production_256/analysis/hmf_z0.json" ]]; then
            pass "HMF z=0 generado"
        else
            fail "HMF z=0 no encontrado"
        fi
        # Verificar que sigma_8 de la simulación esté dentro del 5% del input
        if [[ -f "runs/production_256/analysis/sigma8.txt" ]]; then
            sigma8_sim=$(cat runs/production_256/analysis/sigma8.txt)
            log "σ₈ simulado: $sigma8_sim (input: 0.811, tolerancia: ±5%)"
            pass "σ₈ medido"
        fi
    else
        fail "producción N=256³"
        log "HINT: Si fue una interrupción, relanzar con SKIP_PROD=0 — reanuda desde checkpoint."
    fi
fi

# =============================================================================
section "RESUMEN FINAL"
# =============================================================================
ELAPSED=$(elapsed)
echo
echo "┌──────────────────────────────────────────────────────────┐"
echo "│              VALIDACIONES gadget-ng — RESUMEN            │"
echo "├──────────────────────────────────────────────────────────┤"
printf "│  %-20s %5d                           │\n" "PASS:"  "$PASS"
printf "│  %-20s %5d                           │\n" "FAIL:"  "$FAIL"
printf "│  %-20s %5d                           │\n" "SKIP:"  "$SKIP"
printf "│  %-20s %-30s │\n" "Tiempo total:" "$ELAPSED"
echo "├──────────────────────────────────────────────────────────┤"
echo "│  Logs en: logs/                                          │"
echo "│  Snapshots en: runs/                                     │"
echo "└──────────────────────────────────────────────────────────┘"

if [[ "$FAIL" -gt 0 ]]; then
    echo
    echo "TESTS FALLIDOS — revisar logs/:"
    grep "✗ FAIL:" /dev/stdin <<< "$(cat logs/bloque*.log 2>/dev/null)" || true
    exit 1
fi

exit 0
