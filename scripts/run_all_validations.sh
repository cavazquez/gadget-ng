#!/usr/bin/env bash
# =============================================================================
# run_all_validations.sh — Suite completa de validaciones de gadget-ng
#
# Bloques:
#   Bloque 0: tests lentos --include-ignored (PF-01..16 + pre-existentes)
#             ordenados de más lentos a más rápidos; activar con BLOQUE0_ENABLED=1
#   Bloque 1: cargo test --workspace --release  (todos los tests unitarios)
#   Bloque 2: tests cuantitativos con --nocapture (métricas impresas)
#             Incluye: Phases 161-163 + PF-01..16 (Phase 167)
#   Bloque 3: benchmarks Criterion rápidos
#   Bloque 4: GPU (si hay hardware disponible)
#   Bloque 5: corrida de validación N=128³ (2–4 h, end-to-end cosmológica)
#   Bloque 6: corrida de PRODUCCIÓN N=256³ (8–12 h, la definitiva)
#
# Uso:
#   bash scripts/run_all_validations.sh                         # todo (sin bloque 0)
#   BLOQUE0_ENABLED=1 bash scripts/run_all_validations.sh       # con tests lentos PF
#   SKIP_PROD=1    bash scripts/run_all_validations.sh          # sin N=256³
#   SKIP_VAL128=1  bash scripts/run_all_validations.sh          # sin N=128³
#   ONLY_UNIT=1    bash scripts/run_all_validations.sh          # solo tests rápidos
#   N_RANKS=4      bash scripts/run_all_validations.sh          # MPI×4 para corridas largas
#
# Dejar corriendo toda la noche:
#   nohup BLOQUE0_ENABLED=1 bash scripts/run_all_validations.sh \
#         > logs/validation_$(date +%Y%m%d_%H%M%S).log 2>&1 &
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
section "Bloque 0 — Tests lentos con --include-ignored (PF validations)"
# =============================================================================
# Ordenados de más lentos a más rápidos para maximizar la eficiencia del pipeline.
# Ejecutar solo si no se pide ONLY_UNIT=1.
# Cada test lleva al menos #[ignore] para proteger el CI rápido.

BLOQUE0_ENABLED="${BLOQUE0_ENABLED:-0}"   # 1 = ejecutar bloque 0

if [[ "$BLOQUE0_ENABLED" == "1" && "$ONLY_UNIT" != "1" ]]; then
    # ── Tier 0A: Tests pre-existentes muy lentos (>30 min) ────────────────────
    log "→ [Tier-0A ~2h+] phase42_tree_short_range, phase54, phase55 (--include-ignored)"
    run_test "phase42 (tree short range ignored)" \
        cargo test -p gadget-ng-physics --test phase42_tree_short_range $RELEASE_FLAGS \
        -- --include-ignored 2>&1 | tee logs/bloque0_phase42.log

    run_test "phase54 (growth factor ignored)" \
        cargo test -p gadget-ng-physics --test phase54_growth_factor_validation $RELEASE_FLAGS \
        -- --include-ignored 2>&1 | tee logs/bloque0_phase54.log

    run_test "phase55 (fof vs hmf ignored)" \
        cargo test -p gadget-ng-physics --test phase55_fof_vs_hmf $RELEASE_FLAGS \
        -- --include-ignored 2>&1 | tee logs/bloque0_phase55.log

    # ── Tier 0B: Tests pre-existentes medios (10-30 min) ──────────────────────
    log "→ [Tier-0B ~30m] phase36..41, phase43..44, phase47..49, phase58 (--include-ignored)"
    for t in phase36_pk_correction_validation phase37_growth_rescaled_ics \
              phase38_class_validation phase39_dt_convergence \
              phase40_physical_ics_normalization phase41_high_resolution_validation \
              phase43_dt_treepm_parallel phase44_2lpt_audit \
              phase47_pk_evolution phase48_halofit_validation \
              phase49_halofit_comparison phase58_nfw_concentration; do
        run_test "$t (ignored)" \
            cargo test -p gadget-ng-physics --test "$t" $RELEASE_FLAGS \
            -- --include-ignored 2>&1 | tee "logs/bloque0_${t}.log"
    done

    # ── Tier 0C: PF tests lentos (Tier 1 del plan) ────────────────────────────
    log "→ [Tier-0C ~10min] PF-07 Kolmogorov turbulence spectrum"
    run_test "pf07_mhd_turbulence_spectrum (ignored)" \
        cargo test -p gadget-ng-physics --test pf07_mhd_turbulence_spectrum $RELEASE_FLAGS \
        -- --include-ignored 2>&1 | tee logs/bloque0_pf07.log

    log "→ [Tier-0C ~20min] PF-16 neutrino P(k) suppression sweep"
    run_test "pf16_neutrino_pk_suppression (ignored)" \
        cargo test -p gadget-ng-physics --test pf16_neutrino_pk_suppression $RELEASE_FLAGS \
        -- --include-ignored 2>&1 | tee logs/bloque0_pf16.log

    log "→ [Tier-0C ~5min] PF-05 Sod shock tube Gadget-2"
    run_test "pf05_sod_shock_tube (ignored)" \
        cargo test -p gadget-ng-physics --test pf05_sod_shock_tube $RELEASE_FLAGS \
        -- --include-ignored 2>&1 | tee logs/bloque0_pf05.log

    run_test "gadget2_sph_validation (ignored)" \
        cargo test -p gadget-ng-physics --test gadget2_sph_validation $RELEASE_FLAGS \
        -- --include-ignored 2>&1 | tee logs/bloque0_gadget2_sph.log

    # ── Tier 0D: PF tests medios (Tier 2 del plan) ────────────────────────────
    log "→ [Tier-0D ~5min] PF-04 PM mesh convergence"
    run_test "pf04_pm_mesh_convergence (ignored)" \
        cargo test -p gadget-ng-physics --test pf04_pm_mesh_convergence $RELEASE_FLAGS \
        -- --include-ignored 2>&1 | tee logs/bloque0_pf04.log

    log "→ [Tier-0D ~5min] PF-12 SIDM scatter rate"
    run_test "pf12_sidm_cross_section (ignored)" \
        cargo test -p gadget-ng-physics --test pf12_sidm_cross_section $RELEASE_FLAGS \
        -- --include-ignored 2>&1 | tee logs/bloque0_pf12.log

    log "→ [Tier-0D ~5min] PF-14 Mock SMHM slope"
    run_test "pf14_mock_catalog_smhm (ignored)" \
        cargo test -p gadget-ng-physics --test pf14_mock_catalog_smhm $RELEASE_FLAGS \
        -- --include-ignored 2>&1 | tee logs/bloque0_pf14.log

    log "→ [Tier-0D ~2min] PF-10 two-fluid equilibrium"
    run_test "pf10_two_fluid_equilibrium (ignored)" \
        cargo test -p gadget-ng-physics --test pf10_two_fluid_equilibrium $RELEASE_FLAGS \
        -- --include-ignored 2>&1 | tee logs/bloque0_pf10.log

    log "→ [Tier-0D ~1min] PF-15 X-ray L_X-T_X slope"
    run_test "pf15_xray_lx_tx (ignored)" \
        cargo test -p gadget-ng-physics --test pf15_xray_lx_tx $RELEASE_FLAGS \
        -- --include-ignored 2>&1 | tee logs/bloque0_pf15.log

    log "→ [Tier-0D ~1min] PF-09 RMHD energy conservation"
    run_test "pf09_rmhd_energy_conservation (ignored)" \
        cargo test -p gadget-ng-physics --test pf09_rmhd_energy_conservation $RELEASE_FLAGS \
        -- --include-ignored 2>&1 | tee logs/bloque0_pf09.log

else
    skip "Bloque 0 (tests lentos PF)" "BLOQUE0_ENABLED != 1 o ONLY_UNIT=1  (activar con BLOQUE0_ENABLED=1)"
fi

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
    "phase155_dark_energy_wz"
    "phase156_massive_neutrinos"
    "phase157_sidm"
    "phase158_modified_gravity"
    "phase159_gmc_collapse"
    # Phase 161–163 (HPC validation suite)
    "v3_mhd_validation"
    "v2_hierarchical_cosmo"
    # ── Phase 167: PF validations (tests rápidos sin #[ignore]) ───────────────
    "pf01_leapfrog_convergence"       # PF-01 Leapfrog KDK orden 2
    "pf02_kepler_orbit"               # PF-02 Kepler L + excentricidad
    "pf03_fmm_convergence"            # PF-03 FMM Barnes-Hut θ
    "pf05_sod_shock_tube"             # PF-05 Sod IC checks (lentos: --include-ignored)
    "pf06_sph_pressure_noise"         # PF-06 SPH noise
    "pf07_mhd_turbulence_spectrum"    # PF-07 forcing rápido (lento: --include-ignored)
    "pf08_reconnection_scaling"       # PF-08 Sweet-Parker √η
    "pf09_rmhd_energy_conservation"   # PF-09 RMHD energía (lento: --include-ignored)
    "pf10_two_fluid_equilibrium"      # PF-10 dos fluidos (lento: --include-ignored)
    "pf11_de_luminosity_distance"     # PF-11 energía oscura CPL
    "pf12_sidm_cross_section"         # PF-12 SIDM (rápidos)
    "pf13_fr_chameleon"               # PF-13 f(R) chameleon
    "pf14_mock_catalog_smhm"          # PF-14 SMHM (lento: --include-ignored)
    "pf15_xray_lx_tx"                 # PF-15 L_X-T_X (lento: --include-ignored)
    "pf16_neutrino_pk_suppression"    # PF-16 neutrinos (rápidos)
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
