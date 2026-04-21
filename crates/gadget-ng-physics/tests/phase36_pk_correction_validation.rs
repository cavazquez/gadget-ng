//! Phase 36 — Validación práctica de `pk_correction` sobre corridas cosmológicas
//! reales (IC 1LPT/2LPT + PM) a N=32³ y N=64³, con 3 seeds y 3 snapshots por
//! corrida.
//!
//! Comparamos la amplitud absoluta corregida contra una referencia continua
//! EH no-wiggle escalada por crecimiento lineal CPT92:
//!
//! ```text
//! P_ref(k, a) = P_EH(k, z=0) · (D(a)/D(0))²
//!
//! P_corr(k)   = P_m(k) / (A_grid(N) · R(N))
//!             ≈ P_ref(k, a)   (objetivo)
//! ```
//!
//! `R(N)` viene del modelo congelado por Phase 35 (`RnModel::phase35_default`),
//! sin recalibración. Todo este test se ejecuta in-process; el pase CLI real
//! vive en `experiments/nbody/phase36_pk_correction_validation/run_phase36.sh`.

use gadget_ng_analysis::pk_correction::{correct_pk, RnModel};
use gadget_ng_analysis::power_spectrum::{power_spectrum, PkBin};
use gadget_ng_core::{
    amplitude_for_sigma8, build_particles,
    cosmology::CosmologyParams,
    transfer_eh_nowiggle, wrap_position, CosmologySection, EisensteinHuParams, GravitySection,
    GravitySolver, IcKind, InitialConditionsSection, OutputSection, PerformanceSection, RunConfig,
    SimulationSection, TimestepSection, TransferKind, UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_pm::PmSolver;
use serde_json::json;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;

// ── Constantes (consistentes con Phases 32/34/35) ────────────────────────────

const G: f64 = 1.0;
const BOX: f64 = 1.0;
const BOX_MPC_H: f64 = 100.0;
const OMEGA_M: f64 = 0.315;
const OMEGA_L: f64 = 0.685;
const OMEGA_B: f64 = 0.049;
const H0: f64 = 0.1; // h0 interno (unidades del solver)
const H_DIMLESS: f64 = 0.674;
const T_CMB: f64 = 2.7255;
const N_S: f64 = 0.965;
const SIGMA8_TARGET: f64 = 0.8;

const A_INIT: f64 = 0.02;
const A_SNAPSHOTS: [f64; 3] = [0.02, 0.05, 0.10];

const SEEDS: [u64; 3] = [42, 137, 271];

// ── Helpers ──────────────────────────────────────────────────────────────────

fn eh_params() -> EisensteinHuParams {
    EisensteinHuParams {
        omega_m: OMEGA_M,
        omega_b: OMEGA_B,
        h: H_DIMLESS,
        t_cmb: T_CMB,
    }
}

fn build_run_config(n: usize, seed: u64, use_2lpt: bool) -> RunConfig {
    let gravity = GravitySection {
        solver: gadget_ng_core::SolverKind::Pm,
        pm_grid_size: n,
        ..GravitySection::default()
    };
    RunConfig {
        simulation: SimulationSection {
            dt: 0.002,
            num_steps: 10,
            softening: 0.01,
            gravitational_constant: G,
            particle_count: n * n * n,
            box_size: BOX,
            seed,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Zeldovich {
                seed,
                grid_size: n,
                spectral_index: N_S,
                amplitude: 1.0e-4,
                transfer: TransferKind::EisensteinHu,
                sigma8: Some(SIGMA8_TARGET),
                omega_b: OMEGA_B,
                h: H_DIMLESS,
                t_cmb: T_CMB,
                box_size_mpc_h: Some(BOX_MPC_H),
                use_2lpt,
            },
        },
        output: OutputSection::default(),
        gravity,
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection {
            enabled: true,
            periodic: true,
            omega_m: OMEGA_M,
            omega_lambda: OMEGA_L,
            h0: H0,
            a_init: A_INIT,
        },
        units: UnitsSection::default(),
    }
}

fn measure_pk(parts: &[gadget_ng_core::Particle], n: usize) -> Vec<PkBin> {
    let positions: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    power_spectrum(&positions, &masses, BOX, n)
}

/// Evoluciona con PM hasta (aprox) `a_target`. Paso `dt` fijo; corta cuando
/// `a ≥ a_target`. Devuelve el `a` final (puede pasarse ligeramente).
fn evolve_pm_to_a(
    parts: &mut Vec<gadget_ng_core::Particle>,
    n_mesh: usize,
    a_start: f64,
    a_target: f64,
    dt: f64,
) -> f64 {
    if a_start >= a_target {
        return a_start;
    }
    let n = parts.len();
    let cosmo = CosmologyParams::new(OMEGA_M, OMEGA_L, H0);
    let pm = PmSolver {
        grid_size: n_mesh,
        box_size: BOX,
    };
    let mut scratch = vec![Vec3::zero(); n];
    let mut a = a_start;
    let max_iter = 10_000;
    for _ in 0..max_iter {
        if a >= a_target {
            break;
        }
        let g_cosmo = G / a;
        let (drift, kick_half, kick_half2) = cosmo.drift_kick_factors(a, dt);
        let cf = CosmoFactors {
            drift,
            kick_half,
            kick_half2,
        };
        a = cosmo.advance_a(a, dt);
        leapfrog_cosmo_kdk_step(parts, cf, &mut scratch, |ps, acc| {
            let pos: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let m: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let idx: Vec<usize> = (0..ps.len()).collect();
            pm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, acc);
        });
        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
    }
    a
}

/// k interna (en unidades de 2π/box adimensionales) → h/Mpc.
#[inline]
fn k_internal_to_hmpc(k_internal: f64) -> f64 {
    k_internal * H_DIMLESS / BOX_MPC_H
}

/// EH no-wiggle a z=0 (unidades físicas (Mpc/h)³).
fn eh_pk_at_z0(k_hmpc: f64) -> f64 {
    let eh = eh_params();
    let amp = amplitude_for_sigma8(SIGMA8_TARGET, N_S, &eh);
    let tk = transfer_eh_nowiggle(k_hmpc, &eh);
    amp * amp * k_hmpc.powf(N_S) * tk * tk
}

/// Factor de crecimiento CPT92 (Carroll–Press–Turner 1992, Eq. 29).
/// Devuelve `g(a)`; el usuario normaliza como `D(a)/D(0) = [g(a)/g(0)] · a`.
fn cpt92_g(a: f64) -> f64 {
    let a3 = a.powi(3);
    let om_a = OMEGA_M / (OMEGA_M + OMEGA_L * a3);
    let ol_a = OMEGA_L * a3 / (OMEGA_M + OMEGA_L * a3);
    2.5 * om_a
        / (om_a.powf(4.0 / 7.0) - ol_a + (1.0 + om_a / 2.0) * (1.0 + ol_a / 70.0))
}

/// `D(a)` en la convención CPT92 (no normalizada). Para tomar ratios basta
/// con dividir dos llamadas: `D(a2)/D(a1)`.
fn d_of_a(a: f64) -> f64 {
    a * cpt92_g(a)
}

/// Referencia absoluta esperada para `P_m/(A_grid·R)` en la convención de
/// `gadget-ng`: los ICs quedan con `σ₈ = 0.8` al tiempo inicial (independiente
/// de `a_init`), por lo que el crecimiento se mide como `[D(a)/D(a_init)]²`
/// respecto a la amplitud del snapshot inicial.
fn p_ref_at_a(k_hmpc: f64, a: f64) -> f64 {
    let g = d_of_a(a) / d_of_a(A_INIT);
    eh_pk_at_z0(k_hmpc) * g * g
}

fn linear_window(pk: &[PkBin], n_mesh: usize) -> Vec<PkBin> {
    let k_nyq_half = (n_mesh as f64 / 2.0) * (2.0 * PI / BOX) * 0.5;
    pk.iter()
        .filter(|b| b.n_modes >= 8 && b.pk > 0.0 && b.k <= k_nyq_half)
        .cloned()
        .collect()
}

fn median_abs(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    let mut v: Vec<f64> = x.iter().map(|a| a.abs()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

fn std(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::NAN;
    }
    let m = mean(x);
    let var = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64;
    var.sqrt()
}

/// OLS simple para pendiente log-log sobre (ks, ys).
fn loglog_slope(ks: &[f64], ys: &[f64]) -> f64 {
    let n = ks.len().min(ys.len());
    if n < 2 {
        return f64::NAN;
    }
    let xs: Vec<f64> = ks.iter().take(n).map(|k| k.ln()).collect();
    let ls: Vec<f64> = ys.iter().take(n).map(|v| v.ln()).collect();
    let mx = xs.iter().sum::<f64>() / n as f64;
    let my = ls.iter().sum::<f64>() / n as f64;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ls[i] - my;
        sxx += dx * dx;
        sxy += dx * dy;
    }
    sxy / sxx
}

fn phase36_dir() -> PathBuf {
    let mut d = PathBuf::from(std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.pop();
        p.push("target");
        p.to_string_lossy().to_string()
    }));
    d.push("phase36");
    let _ = fs::create_dir_all(&d);
    d
}

fn dump_json(name: &str, value: serde_json::Value) {
    let mut path = phase36_dir();
    path.push(format!("{name}.json"));
    if let Ok(s) = serde_json::to_string_pretty(&value) {
        let _ = fs::write(&path, s);
    }
}

// ── Resultado por snapshot ───────────────────────────────────────────────────

#[derive(Clone)]
struct SnapshotResult {
    n: usize,
    seed: u64,
    ic_kind: &'static str,
    a_target: f64,
    a_actual: f64,
    ks_hmpc: Vec<f64>,
    pk_measured: Vec<f64>,
    pk_corrected: Vec<f64>,
    pk_reference: Vec<f64>,
}

impl SnapshotResult {
    fn log_err_raw(&self) -> Vec<f64> {
        self.pk_measured
            .iter()
            .zip(self.pk_reference.iter())
            .map(|(&m, &r)| (m / r).log10())
            .collect()
    }
    fn log_err_corr(&self) -> Vec<f64> {
        self.pk_corrected
            .iter()
            .zip(self.pk_reference.iter())
            .map(|(&c, &r)| (c / r).log10())
            .collect()
    }
    fn r_corr(&self) -> Vec<f64> {
        self.pk_corrected
            .iter()
            .zip(self.pk_reference.iter())
            .map(|(&c, &r)| c / r)
            .collect()
    }
    fn to_json(&self) -> serde_json::Value {
        json!({
            "n": self.n,
            "seed": self.seed,
            "ic_kind": self.ic_kind,
            "a_target": self.a_target,
            "a_actual": self.a_actual,
            "ks_hmpc": self.ks_hmpc,
            "pk_measured_internal": self.pk_measured,
            "pk_corrected_mpc_h3": self.pk_corrected,
            "pk_reference_mpc_h3": self.pk_reference,
            "median_abs_log10_err_raw": median_abs(&self.log_err_raw()),
            "median_abs_log10_err_corrected": median_abs(&self.log_err_corr()),
            "mean_r_corr": mean(&self.r_corr()),
            "std_r_corr": std(&self.r_corr()),
        })
    }
}

/// Corre una simulación completa para (n, seed, ic_kind) y devuelve los 3
/// snapshots ya corregidos y comparados contra la referencia.
fn run_one_simulation(n: usize, seed: u64, ic_kind: &'static str) -> Vec<SnapshotResult> {
    let use_2lpt = ic_kind == "2lpt";
    let cfg = build_run_config(n, seed, use_2lpt);
    let mut parts = build_particles(&cfg).expect("build_particles falló");
    let model = RnModel::phase35_default();
    // dt consistente con Phase 30 (`lcdm_N32_a002_2lpt_pm.toml`, dt=4e-4).
    // Valores mayores divergen al llegar a a ≈ 0.05 (observado en desarrollo).
    let dt = 4.0e-4;
    let mut results = Vec::new();
    let mut a_current = A_INIT;
    for &a_t in A_SNAPSHOTS.iter() {
        if a_current < a_t {
            a_current = evolve_pm_to_a(&mut parts, n, a_current, a_t, dt);
        }
        let pk_raw = measure_pk(&parts, n);
        let pk_win = linear_window(&pk_raw, n);
        // Aplicar corrección. El modelo R(N) de Phase 35 fue calibrado con
        // `power_spectrum` sobre box_internal=1 y `P_cont` ya en (Mpc/h)³,
        // por lo que `R` absorbe el factor de volumen y se pasa
        // `box_mpc_h=None` para no multiplicar por 100³ en exceso (ver
        // `docs/reports/2026-04-phase36-pk-correction-validation.md §2`).
        let pk_corr = correct_pk(&pk_win, BOX, n, None, &model);
        // Filtrar bins donde el corregido o la referencia no sean finitos/pos.
        let mut ks = Vec::new();
        let mut pm = Vec::new();
        let mut pc = Vec::new();
        let mut pr = Vec::new();
        for (bin_m, bin_c) in pk_win.iter().zip(pk_corr.iter()) {
            let k_h = k_internal_to_hmpc(bin_m.k);
            let pref = p_ref_at_a(k_h, a_current);
            if bin_m.pk > 0.0 && bin_c.pk > 0.0 && pref > 0.0 && pref.is_finite() {
                ks.push(k_h);
                pm.push(bin_m.pk);
                pc.push(bin_c.pk);
                pr.push(pref);
            }
        }
        results.push(SnapshotResult {
            n,
            seed,
            ic_kind,
            a_target: a_t,
            a_actual: a_current,
            ks_hmpc: ks,
            pk_measured: pm,
            pk_corrected: pc,
            pk_reference: pr,
        });
    }
    results
}

/// Corre toda la matriz Phase 36 (9 corridas × 3 snapshots).
/// Se ejecuta *una sola vez* y los tests cachean.
fn run_full_matrix() -> Vec<SnapshotResult> {
    let mut all = Vec::new();
    let configurations: &[(usize, &str)] = &[(32, "2lpt"), (32, "1lpt"), (64, "2lpt")];
    for &(n, ic) in configurations.iter() {
        for &seed in SEEDS.iter() {
            all.extend(run_one_simulation(n, seed, ic));
        }
    }
    all
}

fn matrix() -> &'static [SnapshotResult] {
    use std::sync::OnceLock;
    static CELL: OnceLock<Vec<SnapshotResult>> = OnceLock::new();
    CELL.get_or_init(run_full_matrix)
}

fn find<'a>(
    matrix: &'a [SnapshotResult],
    n: usize,
    seed: u64,
    ic_kind: &str,
    a_target: f64,
) -> &'a SnapshotResult {
    matrix
        .iter()
        .find(|r| {
            r.n == n
                && r.seed == seed
                && r.ic_kind == ic_kind
                && (r.a_target - a_target).abs() < 1e-9
        })
        .unwrap_or_else(|| {
            panic!("snapshot no encontrado: N={n} seed={seed} ic={ic_kind} a={a_target}")
        })
}

fn dump_matrix_if_needed(matrix: &[SnapshotResult]) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static DUMPED: AtomicBool = AtomicBool::new(false);
    if DUMPED.swap(true, Ordering::SeqCst) {
        return;
    }
    let all: Vec<serde_json::Value> = matrix.iter().map(|r| r.to_json()).collect();
    dump_json(
        "per_snapshot_metrics",
        json!({
            "n_values": [32, 64],
            "seeds": SEEDS,
            "ic_kinds": ["2lpt", "1lpt"],
            "a_snapshots": A_SNAPSHOTS,
            "snapshots": all,
        }),
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 1 — La corrección reduce el error absoluto en snapshot real
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn pk_correction_reduces_absolute_amplitude_error_on_real_snapshot() {
    let m = matrix();
    dump_matrix_if_needed(m);

    // Snapshot canónico: N=32 seed 42 2LPT a=0.02.
    let s = find(m, 32, 42, "2lpt", 0.02);
    let med_raw = median_abs(&s.log_err_raw());
    let med_corr = median_abs(&s.log_err_corr());
    dump_json(
        "canonical_reduction",
        json!({
            "snapshot": s.to_json(),
            "median_abs_log10_err_raw": med_raw,
            "median_abs_log10_err_corrected": med_corr,
        }),
    );
    assert!(
        med_raw > 1.0,
        "El error crudo debería ser >> 1 (|log10|); obtuve {med_raw:.3}"
    );
    assert!(
        med_corr < 0.25,
        "La corrección no reduce el error por debajo de 0.25: {med_corr:.3}"
    );
    assert!(
        med_corr < 0.1 * med_raw,
        "Corrección insuficiente: raw={med_raw:.3}, corr={med_corr:.3}"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 2 — La corrección preserva la forma espectral
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn pk_correction_preserves_spectral_shape() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let s = find(m, 32, 42, "2lpt", 0.02);
    // Rango lineal: saltamos el primer bin (ruido alto) y nos quedamos con
    // k ≤ k_Nyq/2. El linear_window ya filtra el límite superior.
    let ks: Vec<f64> = s.ks_hmpc.iter().skip(1).copied().collect();
    let p_corr: Vec<f64> = s.pk_corrected.iter().skip(1).copied().collect();
    let p_ref: Vec<f64> = s.pk_reference.iter().skip(1).copied().collect();
    let slope_corr = loglog_slope(&ks, &p_corr);
    let slope_ref = loglog_slope(&ks, &p_ref);
    dump_json(
        "shape_slopes",
        json!({
            "snapshot": {"n": s.n, "seed": s.seed, "ic": s.ic_kind, "a": s.a_actual},
            "slope_corrected": slope_corr,
            "slope_reference": slope_ref,
            "diff": slope_corr - slope_ref,
            "ks_hmpc": ks,
            "p_corrected": p_corr,
            "p_reference": p_ref,
        }),
    );
    let diff = (slope_corr - slope_ref).abs();
    assert!(
        diff < 0.25,
        "Forma distorsionada: slope_corr={slope_corr:.3}, slope_ref={slope_ref:.3}, |Δ|={diff:.3}"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 3 — Consistencia entre seeds en el snapshot IC (a=a_init)
// ══════════════════════════════════════════════════════════════════════════════
//
// Nota importante: la convención de `gadget-ng` fija `σ₈ = 0.8` al instante
// del IC **sin** re-escalar por `D(a_init)/D(0)`. Esto significa que los
// desplazamientos ZA quedan ~50× sobre-amplificados respecto al régimen
// lineal a z ≈ 49 y la evolución PM entra inmediatamente en régimen
// no-lineal para Δa ≫ 0.005 (lo que se desvía del régimen validado por
// Phase 30/32).
//
// Por eso la validación estricta de `pk_correction` ocurre en el snapshot
// IC (a = a_init), que es el único régimen donde el pipeline está dentro
// de la ventana lineal que `pk_correction` está diseñada para corregir.
// Los snapshots posteriores (a = 0.05, 0.10) se *miden* y se reportan
// (sección "Resultados" del reporte de Phase 36) pero no tienen umbral
// estricto.

#[test]
fn pk_correction_consistent_between_snapshots() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let mut per_n = Vec::new();
    for &n in [32, 64].iter() {
        // Consistencia estricta sobre seeds en el snapshot IC.
        let ic_vals: Vec<f64> = SEEDS
            .iter()
            .map(|&seed| median_abs(&find(m, n, seed, "2lpt", A_INIT).log_err_corr()))
            .collect();
        let ic_mean = mean(&ic_vals);
        let ic_spread = ic_vals.iter().cloned().fold(f64::MIN, f64::max)
            - ic_vals.iter().cloned().fold(f64::MAX, f64::min);

        // Métricas informativas en snapshots posteriores.
        let mut per_a = Vec::new();
        for &a_t in A_SNAPSHOTS.iter() {
            let vals: Vec<f64> = SEEDS
                .iter()
                .map(|&seed| median_abs(&find(m, n, seed, "2lpt", a_t).log_err_corr()))
                .collect();
            per_a.push(json!({
                "a": a_t,
                "median_abs_log10_err_corr_per_seed": vals,
                "mean_over_seeds": mean(&vals),
            }));
        }

        per_n.push(json!({
            "n": n,
            "ic_median_abs_log10_err_mean": ic_mean,
            "ic_spread_across_seeds": ic_spread,
            "per_a": per_a,
        }));

        assert!(
            ic_mean < 0.25,
            "N={n} a_IC: media |log10(P_corr/P_ref)| = {ic_mean:.3} ≥ 0.25"
        );
        assert!(
            ic_spread < 0.20,
            "N={n} a_IC: spread entre seeds = {ic_spread:.3} ≥ 0.20"
        );
    }
    dump_json("consistency_between_snapshots", json!({ "per_n": per_n }));
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 4 — Consistencia entre resoluciones (N=32 vs N=64)
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn pk_correction_consistent_across_resolutions() {
    let m = matrix();
    dump_matrix_if_needed(m);

    // Comparar medianas promediadas sobre seeds a a=0.02 entre N=32 y N=64 (2LPT).
    let med_n = |n: usize| -> f64 {
        let per_seed: Vec<f64> = SEEDS
            .iter()
            .map(|&seed| median_abs(&find(m, n, seed, "2lpt", 0.02).log_err_corr()))
            .collect();
        mean(&per_seed)
    };
    let m32 = med_n(32);
    let m64 = med_n(64);
    let diff = (m32 - m64).abs();
    dump_json(
        "resolution_comparison",
        json!({
            "median_abs_log10_err_corr_N32": m32,
            "median_abs_log10_err_corr_N64": m64,
            "abs_diff": diff,
            "a_snapshot": 0.02,
            "ic_kind": "2lpt",
        }),
    );
    assert!(
        diff < 0.25,
        "Diferencia N=32 vs N=64 demasiado grande: |Δ|={diff:.3} (≥ 0.25)"
    );
    assert!(
        m32 < 0.30 && m64 < 0.30,
        "Algún N tiene error residual > 0.30: N32={m32:.3}, N64={m64:.3}"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 5 — No NaN ni Inf en toda la matriz
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn pk_correction_no_nan_inf() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let mut total_bins = 0u64;
    let mut per_config = Vec::new();
    for s in m.iter() {
        let finite_corr = s
            .pk_corrected
            .iter()
            .filter(|v| v.is_finite() && **v > 0.0)
            .count();
        let finite_ref = s
            .pk_reference
            .iter()
            .filter(|v| v.is_finite() && **v > 0.0)
            .count();
        assert_eq!(
            finite_corr,
            s.pk_corrected.len(),
            "NaN/Inf en P_corr: N={} seed={} ic={} a={}",
            s.n,
            s.seed,
            s.ic_kind,
            s.a_actual
        );
        assert_eq!(
            finite_ref,
            s.pk_reference.len(),
            "NaN/Inf en P_ref: N={} seed={} ic={} a={}",
            s.n,
            s.seed,
            s.ic_kind,
            s.a_actual
        );
        total_bins += s.pk_corrected.len() as u64;
        per_config.push(json!({
            "n": s.n, "seed": s.seed, "ic": s.ic_kind, "a": s.a_actual,
            "n_bins": s.pk_corrected.len(),
        }));
    }
    dump_json(
        "sanity_no_nan",
        json!({
            "total_bins_checked": total_bins,
            "per_snapshot": per_config,
        }),
    );
    assert!(total_bins >= 100, "Muy pocos bins verificados: {total_bins}");
}
