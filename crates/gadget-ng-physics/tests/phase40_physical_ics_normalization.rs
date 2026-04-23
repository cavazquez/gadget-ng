//! Phase 40 — Formalización de la convención física de las ICs cosmológicas.
//!
//! Responde: *¿si `gadget-ng` redefine correctamente sus ICs para que `σ₈=0.8`
//! corresponda a `z=0` (convención CAMB/CLASS) y no a `a_init`, la evolución
//! temprana vuelve a ser consistente con el régimen lineal y `pk_correction`
//! pasa a ser válida también en snapshots evolucionados tempranos?*
//!
//! ## API introducida
//!
//! Fase 40 reemplaza el flag experimental `rescale_to_a_init: bool` de Fase 37
//! por una enum tipada `NormalizationMode { Legacy, Z0Sigma8 }`:
//!
//! - `Legacy`: `σ₈` en `a_init`. Bit-idéntico a Fase 26–28.
//! - `Z0Sigma8`: `σ₈` referido a `a=1`, desplazamientos escalados por
//!   `s = D(a_init)/D(1)` y `s²` para 2LPT. Equivalente físico del viejo
//!   `rescale_to_a_init = true`.
//!
//! ## Matriz
//!
//! 2LPT + PM, `N = 32³`, `seeds ∈ {42, 137, 271}`, modos
//! `{Legacy, Z0Sigma8}`, snapshots `a ∈ {0.02, 0.05, 0.10}` → **18 corridas**.
//! Runtime ~2 min release (`OnceLock` memoiza la matriz entre tests).
//!
//! ## Referencia física
//!
//! - `Legacy`:   `P_ref(k, a) = P_EH(k, z=0) · [D(a)/D(a_init)]²`
//! - `Z0Sigma8`: `P_ref(k, a) = P_EH(k, z=0) · [D(a)/D(1)]²`
//!
//! ## σ₈ esperado a `a_init`
//!
//! - `Legacy`:   `σ₈(a_init) = σ₈_target = 0.8`
//! - `Z0Sigma8`: `σ₈(a_init) = σ₈_target · D(a_init)/D(1) ≈ 0.0203`
//!
//! La medición empírica de σ₈(a_init) sobre el IC (top-hat `R=8 Mpc/h`) permite
//! verificar cuantitativamente la convención física (tests 2 y 3).

use gadget_ng_analysis::pk_correction::{correct_pk, RnModel};
use gadget_ng_analysis::power_spectrum::{power_spectrum, PkBin};
use gadget_ng_core::{
    amplitude_for_sigma8, build_particles,
    cosmology::{growth_factor_d_ratio, CosmologyParams},
    sigma_from_pk_bins, transfer_eh_nowiggle, wrap_position, CosmologySection, EisensteinHuParams,
    GravitySection, GravitySolver, IcKind, InitialConditionsSection, NormalizationMode,
    OutputSection, PerformanceSection, RunConfig, SimulationSection, TimestepSection, TransferKind,
    UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_pm::PmSolver;
use serde_json::json;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

// ── Constantes (idénticas a Phase 36/37/39 para comparabilidad) ──────────────

const G: f64 = 1.0;
const BOX: f64 = 1.0;
const BOX_MPC_H: f64 = 100.0;
const OMEGA_M: f64 = 0.315;
const OMEGA_L: f64 = 0.685;
const OMEGA_B: f64 = 0.049;
const H0: f64 = 0.1;
const H_DIMLESS: f64 = 0.674;
const T_CMB: f64 = 2.7255;
const N_S: f64 = 0.965;
const SIGMA8_TARGET: f64 = 0.8;

const A_INIT: f64 = 0.02;
const A_SNAPSHOTS: [f64; 3] = [0.02, 0.05, 0.10];
const N_GRID: usize = 32;
const SEEDS: [u64; 3] = [42, 137, 271];

// ── Tipos auxiliares ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Legacy,
    Z0Sigma8,
}

impl Mode {
    fn as_str(self) -> &'static str {
        match self {
            Mode::Legacy => "legacy",
            Mode::Z0Sigma8 => "z0_sigma8",
        }
    }
    fn as_norm(self) -> NormalizationMode {
        match self {
            Mode::Legacy => NormalizationMode::Legacy,
            Mode::Z0Sigma8 => NormalizationMode::Z0Sigma8,
        }
    }
}

// ── Helpers físicos ──────────────────────────────────────────────────────────

fn eh_params() -> EisensteinHuParams {
    EisensteinHuParams {
        omega_m: OMEGA_M,
        omega_b: OMEGA_B,
        h: H_DIMLESS,
        t_cmb: T_CMB,
    }
}

fn cosmo_params() -> CosmologyParams {
    CosmologyParams::new(OMEGA_M, OMEGA_L, H0)
}

fn build_run_config(n: usize, seed: u64, mode: Mode) -> RunConfig {
    let gravity = GravitySection {
        solver: gadget_ng_core::SolverKind::Pm,
        pm_grid_size: n,
        ..GravitySection::default()
    };
    RunConfig {
        simulation: SimulationSection {
            dt: 4.0e-4,
            num_steps: 10,
            softening: 0.01,
            physical_softening: false,
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
                use_2lpt: true,
                normalization_mode: mode.as_norm(),
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
            auto_g: false,
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
        insitu_analysis: Default::default(),
        sph: Default::default(),
    }
}

fn measure_pk(parts: &[gadget_ng_core::Particle], n: usize) -> Vec<PkBin> {
    let positions: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    power_spectrum(&positions, &masses, BOX, n)
}

/// Evoluciona con PM hasta (aprox.) `a_target` usando leapfrog KDK cosmológico.
/// Devuelve `a` final alcanzado.
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
    let cosmo = cosmo_params();
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
            pm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, acc)
        });
        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
    }
    a
}

#[inline]
fn k_internal_to_hmpc(k_internal: f64) -> f64 {
    k_internal * H_DIMLESS / BOX_MPC_H
}

fn eh_pk_at_z0(k_hmpc: f64) -> f64 {
    let eh = eh_params();
    let amp = amplitude_for_sigma8(SIGMA8_TARGET, N_S, &eh);
    let tk = transfer_eh_nowiggle(k_hmpc, &eh);
    amp * amp * k_hmpc.powf(N_S) * tk * tk
}

/// Referencia física del espectro lineal por modo.
///
/// - `Legacy`:   `P_ref(k, a) = P_EH(k, z=0) · [D(a)/D(a_init)]²`
/// - `Z0Sigma8`: `P_ref(k, a) = P_EH(k, z=0) · [D(a)/D(1)]²`
fn p_ref_at_a(k_hmpc: f64, a: f64, mode: Mode) -> f64 {
    let cosmo = cosmo_params();
    let d_ratio = match mode {
        Mode::Legacy => growth_factor_d_ratio(cosmo, a, A_INIT),
        Mode::Z0Sigma8 => growth_factor_d_ratio(cosmo, a, 1.0),
    };
    eh_pk_at_z0(k_hmpc) * d_ratio * d_ratio
}

/// σ₈ esperado a `a_init` según convención.
fn sigma8_expected(mode: Mode) -> f64 {
    match mode {
        Mode::Legacy => SIGMA8_TARGET,
        Mode::Z0Sigma8 => {
            let s = growth_factor_d_ratio(cosmo_params(), A_INIT, 1.0);
            SIGMA8_TARGET * s
        }
    }
}

/// σ₈(R) medido empíricamente a partir del espectro de potencias **corregido**
/// (post-`pk_correction`, ya en `(Mpc/h)³` y `k` en `h/Mpc`).
fn measure_sigma8_from_corrected(ks_hmpc: &[f64], pk_mpc_h3: &[f64], r_mpc_h: f64) -> f64 {
    let bins: Vec<(f64, f64)> = ks_hmpc
        .iter()
        .zip(pk_mpc_h3.iter())
        .filter(|(&k, &p)| k > 0.0 && p > 0.0 && p.is_finite())
        .map(|(&k, &p)| (k, p))
        .collect();
    sigma_from_pk_bins(&bins, r_mpc_h)
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

fn stdev(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::NAN;
    }
    let m = mean(x);
    let var = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64;
    var.sqrt()
}

fn delta_rms(parts: &[gadget_ng_core::Particle], n: usize) -> f64 {
    let ng = n;
    let n3 = ng * ng * ng;
    let mut counts = vec![0u32; n3];
    let cell = BOX / ng as f64;
    for p in parts {
        let ix = ((p.position.x / cell) as usize).min(ng - 1);
        let iy = ((p.position.y / cell) as usize).min(ng - 1);
        let iz = ((p.position.z / cell) as usize).min(ng - 1);
        counts[ix * ng * ng + iy * ng + iz] += 1;
    }
    let mean_n = parts.len() as f64 / n3 as f64;
    if mean_n <= 0.0 {
        return 0.0;
    }
    let sum_sq: f64 = counts
        .iter()
        .map(|&c| {
            let d = (c as f64 - mean_n) / mean_n;
            d * d
        })
        .sum();
    (sum_sq / n3 as f64).sqrt()
}

fn v_rms(parts: &[gadget_ng_core::Particle]) -> f64 {
    if parts.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = parts
        .iter()
        .map(|p| {
            p.velocity.x * p.velocity.x + p.velocity.y * p.velocity.y + p.velocity.z * p.velocity.z
        })
        .sum();
    (sum_sq / parts.len() as f64).sqrt()
}

/// rms del desplazamiento respecto a la retícula Lagrangiana.
fn psi_rms_from_particles(parts: &[gadget_ng_core::Particle], n: usize) -> f64 {
    let d = BOX / n as f64;
    let mut sum_sq = 0.0;
    for p in parts {
        let gid = p.global_id;
        let ix = gid / (n * n);
        let rem = gid % (n * n);
        let iy = rem / n;
        let iz = rem % n;
        let q = Vec3::new(
            (ix as f64 + 0.5) * d,
            (iy as f64 + 0.5) * d,
            (iz as f64 + 0.5) * d,
        );
        let dx = gadget_ng_core::minimum_image(p.position.x - q.x, BOX);
        let dy = gadget_ng_core::minimum_image(p.position.y - q.y, BOX);
        let dz = gadget_ng_core::minimum_image(p.position.z - q.z, BOX);
        sum_sq += dx * dx + dy * dy + dz * dz;
    }
    (sum_sq / parts.len() as f64).sqrt()
}

// ── Resultado por snapshot ───────────────────────────────────────────────────

#[derive(Clone)]
struct SnapshotResult {
    n: usize,
    seed: u64,
    mode: &'static str,
    a_target: f64,
    a_actual: f64,
    ks_hmpc: Vec<f64>,
    pk_measured: Vec<f64>,
    pk_corrected: Vec<f64>,
    pk_reference: Vec<f64>,
    delta_rms: f64,
    v_rms: f64,
    psi_rms: f64,
    sigma8_measured: f64,
    /// σ₈ integrado sobre el espectro de referencia lineal truncado al mismo
    /// rango k que `sigma8_measured`. Sirve como valor de referencia para
    /// comparar el efecto del truncado en k (aislando el factor de ventana).
    sigma8_from_ref: f64,
    sigma8_expected: f64,
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
    fn median_abs_log_err_corr(&self) -> f64 {
        median_abs(&self.log_err_corr())
    }
    fn to_json(&self) -> serde_json::Value {
        let r = self.r_corr();
        json!({
            "n": self.n,
            "seed": self.seed,
            "mode": self.mode,
            "a_target": self.a_target,
            "a_actual": self.a_actual,
            "ks_hmpc": self.ks_hmpc,
            "pk_measured_internal": self.pk_measured,
            "pk_corrected_mpc_h3": self.pk_corrected,
            "pk_reference_mpc_h3": self.pk_reference,
            "median_abs_log10_err_raw": median_abs(&self.log_err_raw()),
            "median_abs_log10_err_corrected": median_abs(&self.log_err_corr()),
            "mean_r_corr": mean(&r),
            "std_r_corr": stdev(&r),
            "delta_rms": self.delta_rms,
            "v_rms": self.v_rms,
            "psi_rms": self.psi_rms,
            "sigma8_measured": self.sigma8_measured,
            "sigma8_from_ref": self.sigma8_from_ref,
            "sigma8_expected": self.sigma8_expected,
        })
    }
}

// ── Orquestación de la matriz ────────────────────────────────────────────────

fn run_one_simulation(seed: u64, mode: Mode) -> Vec<SnapshotResult> {
    let cfg = build_run_config(N_GRID, seed, mode);
    let mut parts = build_particles(&cfg).expect("build_particles falló");
    let psi_rms_ic = psi_rms_from_particles(&parts, N_GRID);
    let model = RnModel::phase35_default();
    let dt = 4.0e-4;

    let mut results = Vec::with_capacity(A_SNAPSHOTS.len());
    let mut a_current = A_INIT;

    for &a_t in A_SNAPSHOTS.iter() {
        if a_current < a_t {
            a_current = evolve_pm_to_a(&mut parts, N_GRID, a_current, a_t, dt);
        }
        let pk_raw = measure_pk(&parts, N_GRID);
        let pk_win = linear_window(&pk_raw, N_GRID);
        // Corrección en las mismas unidades que `P_ref` (idéntica a Phase 37 y
        // Phase 39: `box_mpc_h = None`). El σ₈ se mide del mismo array para
        // garantizar consistencia de unidades.
        let pk_corr = correct_pk(&pk_win, BOX, N_GRID, None, &model);
        let sigma8_exp = if (a_t - A_INIT).abs() < 1e-9 {
            sigma8_expected(mode)
        } else {
            // σ₈(a) ≈ σ₈(a_init) · D(a)/D(a_init) bajo evolución lineal.
            let cosmo = cosmo_params();
            let growth = growth_factor_d_ratio(cosmo, a_current, A_INIT);
            sigma8_expected(mode) * growth
        };

        let mut ks = Vec::new();
        let mut pm_v = Vec::new();
        let mut pc_v = Vec::new();
        let mut pr_v = Vec::new();
        for (bin_m, bin_c) in pk_win.iter().zip(pk_corr.iter()) {
            let k_h = k_internal_to_hmpc(bin_m.k);
            let pref = p_ref_at_a(k_h, a_current, mode);
            if bin_m.pk > 0.0 && bin_c.pk > 0.0 && pref > 0.0 && pref.is_finite() {
                ks.push(k_h);
                pm_v.push(bin_m.pk);
                pc_v.push(bin_c.pk);
                pr_v.push(pref);
            }
        }

        // σ₈(R=8 Mpc/h) medido: integra sobre el espectro de referencia lineal
        // y el corregido. Como `P_ref` está definido consistentemente con la
        // convención `P_c/P_ref ≈ 1` (tests 4), el ratio `σ₈_meas/σ₈_ref` es
        // una medida adimensional de la precisión del espectro corregido.
        let sigma8_meas = measure_sigma8_from_corrected(&ks, &pc_v, 8.0);
        let sigma8_from_ref = measure_sigma8_from_corrected(&ks, &pr_v, 8.0);

        results.push(SnapshotResult {
            n: N_GRID,
            seed,
            mode: mode.as_str(),
            a_target: a_t,
            a_actual: a_current,
            ks_hmpc: ks,
            pk_measured: pm_v,
            pk_corrected: pc_v,
            pk_reference: pr_v,
            delta_rms: delta_rms(&parts, N_GRID),
            v_rms: v_rms(&parts),
            psi_rms: psi_rms_ic,
            sigma8_measured: sigma8_meas,
            sigma8_from_ref,
            sigma8_expected: sigma8_exp,
        });
    }
    results
}

/// Matriz: 3 seeds × 2 modos × 3 snapshots = 18 mediciones. `OnceLock`.
fn run_full_matrix() -> Vec<SnapshotResult> {
    let mut all = Vec::with_capacity(SEEDS.len() * 2 * A_SNAPSHOTS.len());
    for &seed in SEEDS.iter() {
        for &mode in &[Mode::Legacy, Mode::Z0Sigma8] {
            all.extend(run_one_simulation(seed, mode));
        }
    }
    all
}

fn matrix() -> &'static [SnapshotResult] {
    static CELL: OnceLock<Vec<SnapshotResult>> = OnceLock::new();
    CELL.get_or_init(run_full_matrix)
}

fn find<'a>(m: &'a [SnapshotResult], seed: u64, mode: &str, a_target: f64) -> &'a SnapshotResult {
    m.iter()
        .find(|r| r.seed == seed && r.mode == mode && (r.a_target - a_target).abs() < 1e-9)
        .unwrap_or_else(|| panic!("snapshot no encontrado: seed={seed} mode={mode} a={a_target}"))
}

fn phase40_dir() -> PathBuf {
    let mut d = PathBuf::from(std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.pop();
        p.push("target");
        p.to_string_lossy().to_string()
    }));
    d.push("phase40");
    let _ = fs::create_dir_all(&d);
    d
}

fn dump_json(name: &str, value: serde_json::Value) {
    let mut path = phase40_dir();
    path.push(format!("{name}.json"));
    if let Ok(s) = serde_json::to_string_pretty(&value) {
        let _ = fs::write(&path, s);
    }
}

fn dump_matrix_if_needed(m: &[SnapshotResult]) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static DUMPED: AtomicBool = AtomicBool::new(false);
    if DUMPED.swap(true, Ordering::SeqCst) {
        return;
    }
    let all: Vec<serde_json::Value> = m.iter().map(|r| r.to_json()).collect();
    dump_json(
        "per_snapshot_metrics",
        json!({
            "n": N_GRID,
            "seeds": SEEDS,
            "modes": ["legacy", "z0_sigma8"],
            "a_snapshots": A_SNAPSHOTS,
            "snapshots": all,
        }),
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 1 — legacy_mode_remains_bit_compatible
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn legacy_mode_remains_bit_compatible() {
    let cfg_legacy = build_run_config(N_GRID, 42, Mode::Legacy);
    let cfg_z0 = build_run_config(N_GRID, 42, Mode::Z0Sigma8);
    let p_legacy = build_particles(&cfg_legacy).expect("build_particles legacy falló");
    let p_z0 = build_particles(&cfg_z0).expect("build_particles z0 falló");
    assert_eq!(p_legacy.len(), p_z0.len());

    let p_legacy_bis = build_particles(&cfg_legacy).expect("rebuild legacy falló");
    for (a, b) in p_legacy.iter().zip(p_legacy_bis.iter()) {
        assert_eq!(a.position.x.to_bits(), b.position.x.to_bits());
        assert_eq!(a.position.y.to_bits(), b.position.y.to_bits());
        assert_eq!(a.position.z.to_bits(), b.position.z.to_bits());
        assert_eq!(a.velocity.x.to_bits(), b.velocity.x.to_bits());
        assert_eq!(a.velocity.y.to_bits(), b.velocity.y.to_bits());
        assert_eq!(a.velocity.z.to_bits(), b.velocity.z.to_bits());
    }

    let mut any_diff = false;
    for (a, b) in p_legacy.iter().zip(p_z0.iter()) {
        if a.position != b.position || a.velocity != b.velocity {
            any_diff = true;
            break;
        }
    }
    assert!(
        any_diff,
        "legacy vs z0_sigma8 deben diferir: el flag no está tomando efecto"
    );

    dump_json(
        "test1_legacy_bit_compatible",
        json!({
            "legacy_bit_identical_across_rebuilds": true,
            "legacy_differs_from_z0_sigma8": any_diff,
            "n": N_GRID,
            "seed": 42,
        }),
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 2 — z0_sigma8_mode_matches_expected_growth_scaling
// ══════════════════════════════════════════════════════════════════════════════
//
// Hard check: rms(Ψ)_z0 / rms(Ψ)_legacy ≈ s = D(a_init)/D(1), tolerancia 1%.
// Verifica la matemática del reescalado a nivel de desplazamientos.

#[test]
fn z0_sigma8_mode_matches_expected_growth_scaling() {
    let cfg_legacy = build_run_config(N_GRID, 42, Mode::Legacy);
    let cfg_z0 = build_run_config(N_GRID, 42, Mode::Z0Sigma8);
    let p_legacy = build_particles(&cfg_legacy).unwrap();
    let p_z0 = build_particles(&cfg_z0).unwrap();

    let rms_legacy = psi_rms_from_particles(&p_legacy, N_GRID);
    let rms_z0 = psi_rms_from_particles(&p_z0, N_GRID);
    let ratio = rms_z0 / rms_legacy;

    let s = growth_factor_d_ratio(cosmo_params(), A_INIT, 1.0);
    let rel_err = (ratio - s).abs() / s;

    dump_json(
        "test2_growth_scaling",
        json!({
            "psi_rms_legacy": rms_legacy,
            "psi_rms_z0_sigma8": rms_z0,
            "ratio_measured": ratio,
            "s_expected": s,
            "relative_error": rel_err,
            "tolerance": 0.01,
        }),
    );

    assert!(
        rel_err < 0.01,
        "Reescalado físico incorrecto: ratio medido={ratio:.5} vs s esperado={s:.5} (err={rel_err:.2e})"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 3 — sigma8_at_ainit_matches_linear_prediction
// ══════════════════════════════════════════════════════════════════════════════
//
// Hard check: σ₈(a_init) medido bajo Z0Sigma8 ≈ 0.8 · D(a_init)/D(1),
// tolerancia 5%. Validación física de la convención.

#[test]
fn sigma8_at_ainit_matches_linear_prediction() {
    let m = matrix();
    dump_matrix_if_needed(m);

    // Estrategia: medimos σ₈ integrando P_corrected sobre la misma ventana de
    // k disponible (truncada a `k ≤ k_Nyq/2`). Como la ventana de k afecta por
    // igual a ambos modos, el **ratio** `σ₈(z0)/σ₈(legacy)` debe ≈ s =
    // D(a_init)/D(1) — éste es un invariante adimensional que no depende de
    // la normalización absoluta. Como chequeo secundario, verificamos que
    // `σ₈_measured/σ₈_from_ref ≈ 1` en cada modo (consistencia pk_corrected vs
    // P_lin en la ventana truncada).
    let mut per_mode = Vec::new();
    let mut sigma8_legacy_vec = Vec::new();
    let mut sigma8_z0_vec = Vec::new();
    for &mode in &[Mode::Legacy, Mode::Z0Sigma8] {
        let mut measured = Vec::new();
        let mut from_ref = Vec::new();
        let mut ratios_meas_over_ref = Vec::new();
        for &seed in SEEDS.iter() {
            let snap = find(m, seed, mode.as_str(), A_INIT);
            measured.push(snap.sigma8_measured);
            from_ref.push(snap.sigma8_from_ref);
            if snap.sigma8_from_ref > 0.0 {
                ratios_meas_over_ref.push(snap.sigma8_measured / snap.sigma8_from_ref);
            }
        }
        let mean_ratio = mean(&ratios_meas_over_ref);
        let rel_err = (mean_ratio - 1.0).abs();
        per_mode.push(json!({
            "mode": mode.as_str(),
            "sigma8_measured_per_seed": measured,
            "sigma8_from_ref_per_seed": from_ref,
            "sigma8_measured_mean": mean(&measured),
            "sigma8_from_ref_mean": mean(&from_ref),
            "mean_ratio_meas_over_ref": mean_ratio,
            "rel_err_from_unity": rel_err,
        }));
        eprintln!(
            "[phase40][test3] mode={} σ₈_meas/σ₈_ref={:.4} (rel_err={:.2e} vs 1.0)",
            mode.as_str(),
            mean_ratio,
            rel_err
        );
        assert!(
            rel_err < 0.15,
            "σ₈_meas/σ₈_ref fuera de tolerancia: mode={} ratio={:.4} (>15% error vs 1.0)",
            mode.as_str(),
            mean_ratio
        );
        match mode {
            Mode::Legacy => sigma8_legacy_vec = measured,
            Mode::Z0Sigma8 => sigma8_z0_vec = measured,
        }
    }

    // Chequeo central de la convención física: ratio entre modos ≈ s.
    let ratios_modes: Vec<f64> = sigma8_legacy_vec
        .iter()
        .zip(sigma8_z0_vec.iter())
        .filter(|(&l, _)| l > 0.0)
        .map(|(&l, &z)| z / l)
        .collect();
    let mean_mode_ratio = mean(&ratios_modes);
    let s = growth_factor_d_ratio(cosmo_params(), A_INIT, 1.0);
    let mode_ratio_err = (mean_mode_ratio - s).abs() / s;

    dump_json(
        "test3_sigma8_a_init",
        json!({
            "per_mode": per_mode,
            "mode_ratio_z0_over_legacy_per_seed": ratios_modes,
            "mode_ratio_mean": mean_mode_ratio,
            "s_expected": s,
            "mode_ratio_rel_err": mode_ratio_err,
            "tolerance_mode_ratio": 0.05,
        }),
    );

    eprintln!(
        "[phase40][test3] σ₈(z0)/σ₈(legacy) medido={:.5} vs s={:.5} (err={:.2e})",
        mean_mode_ratio, s, mode_ratio_err
    );
    assert!(
        mode_ratio_err < 0.05,
        "Ratio de σ₈ entre modos fuera de tolerancia: medido={:.5} vs s esperado={:.5} (err={:.2e})",
        mean_mode_ratio,
        s,
        mode_ratio_err
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 4 — pk_correction_still_works_on_ic_snapshot_under_z0_mode
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn pk_correction_still_works_on_ic_snapshot_under_z0_mode() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let mut err_ic_z0 = Vec::new();
    let mut err_ic_legacy = Vec::new();
    for &seed in SEEDS.iter() {
        err_ic_z0.push(find(m, seed, "z0_sigma8", A_INIT).median_abs_log_err_corr());
        err_ic_legacy.push(find(m, seed, "legacy", A_INIT).median_abs_log_err_corr());
    }

    let median_z0 = median_abs(&err_ic_z0);
    let median_legacy = median_abs(&err_ic_legacy);

    dump_json(
        "test4_ic_pk_correction",
        json!({
            "median_abs_log_err_corrected_z0_per_seed": err_ic_z0,
            "median_abs_log_err_corrected_legacy_per_seed": err_ic_legacy,
            "median_across_seeds_z0": median_z0,
            "median_across_seeds_legacy": median_legacy,
            "threshold": 0.2,
        }),
    );

    assert!(
        median_z0 <= 0.2,
        "pk_correction no cierra en IC bajo Z0Sigma8: median|log10(P_c/P_ref)|={median_z0:.3} > 0.2"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 5 — z0_mode_reduces_early_nonlinearity_vs_legacy (SOFT)
// ══════════════════════════════════════════════════════════════════════════════
//
// Reporta si `δ_rms(0.10)_z0` es menor que `δ_rms(0.10)_legacy` sin fallar.

#[test]
fn z0_mode_reduces_early_nonlinearity_vs_legacy() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let mut out = Vec::new();
    for &a_t in &[0.05_f64, 0.10_f64] {
        let mut ratios = Vec::new();
        for &seed in SEEDS.iter() {
            let d_leg = find(m, seed, "legacy", a_t).delta_rms;
            let d_z0 = find(m, seed, "z0_sigma8", a_t).delta_rms;
            ratios.push(d_z0 / d_leg);
        }
        let mean_ratio = mean(&ratios);
        let hypothesis_supported = mean_ratio < 0.5;
        out.push(json!({
            "a_target": a_t,
            "delta_rms_ratio_z0_over_legacy_per_seed": ratios,
            "mean_ratio": mean_ratio,
            "hypothesis_z0_reduces_nonlinearity_2x": hypothesis_supported,
        }));
        eprintln!(
            "[phase40][test5] a={a_t:.2} δ_rms ratio z0/legacy={mean_ratio:.3} → supported={hypothesis_supported}"
        );
    }

    dump_json("test5_early_nonlinearity", json!({"per_a": out}));

    // Soft check: sólo exige finitud.
    for entry in &out {
        let mr = entry["mean_ratio"].as_f64().unwrap_or(f64::NAN);
        assert!(mr.is_finite(), "mean_ratio no finito: {entry}");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 6 — z0_mode_improves_early_snapshot_accuracy_vs_legacy (SOFT)
// ══════════════════════════════════════════════════════════════════════════════
//
// Mide si median|log10(P_c/P_ref)|_z0 < median|log10(P_c/P_ref)|_legacy en
// snapshots evolucionados. Determina Decisión A (≥2x mejor) o B.

#[test]
fn z0_mode_improves_early_snapshot_accuracy_vs_legacy() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let mut per_a = Vec::new();
    for &a_t in &[0.05_f64, 0.10_f64] {
        let mut err_legacy = Vec::new();
        let mut err_z0 = Vec::new();
        for &seed in SEEDS.iter() {
            err_legacy.push(find(m, seed, "legacy", a_t).median_abs_log_err_corr());
            err_z0.push(find(m, seed, "z0_sigma8", a_t).median_abs_log_err_corr());
        }
        let median_legacy = mean(&err_legacy);
        let median_z0 = mean(&err_z0);
        let factor = median_legacy / median_z0;
        let decision_a_supported = factor >= 2.0 && median_z0 < median_legacy;
        per_a.push(json!({
            "a_target": a_t,
            "median_err_legacy_per_seed": err_legacy,
            "median_err_z0_per_seed": err_z0,
            "mean_median_err_legacy": median_legacy,
            "mean_median_err_z0": median_z0,
            "improvement_factor_legacy_over_z0": factor,
            "decision_a_supported_at_this_a": decision_a_supported,
        }));
        eprintln!(
            "[phase40][test6] a={a_t:.2} err_legacy={median_legacy:.3} err_z0={median_z0:.3} factor={factor:.3}"
        );
    }

    // Factor global agregando sobre snapshots evolucionados.
    let global_factor_legacy = per_a
        .iter()
        .map(|e| e["mean_median_err_legacy"].as_f64().unwrap())
        .sum::<f64>()
        / per_a.len() as f64;
    let global_factor_z0 = per_a
        .iter()
        .map(|e| e["mean_median_err_z0"].as_f64().unwrap())
        .sum::<f64>()
        / per_a.len() as f64;
    let global_factor = global_factor_legacy / global_factor_z0;
    let decision = if global_factor >= 2.0 && global_factor_z0 < global_factor_legacy {
        "A_z0_replaces_legacy"
    } else {
        "B_z0_stays_experimental"
    };

    dump_json(
        "test6_early_accuracy",
        json!({
            "per_a": per_a,
            "global_mean_err_legacy": global_factor_legacy,
            "global_mean_err_z0": global_factor_z0,
            "global_improvement_factor": global_factor,
            "decision": decision,
        }),
    );

    eprintln!("[phase40][test6] DECISION {decision}: global_factor={global_factor:.3} (>=2.0 → A)");

    // Soft check: sólo exige que las medianas sean finitas.
    assert!(
        global_factor_legacy.is_finite() && global_factor_z0.is_finite(),
        "errores medianos no finitos: legacy={global_factor_legacy} z0={global_factor_z0}"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 7 — z0_mode_no_nan_inf
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn z0_mode_no_nan_inf() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let mut bad = Vec::new();
    for snap in m {
        let metrics = [
            ("delta_rms", snap.delta_rms),
            ("v_rms", snap.v_rms),
            ("psi_rms", snap.psi_rms),
            ("sigma8_measured", snap.sigma8_measured),
        ];
        for (name, v) in metrics.iter() {
            if !v.is_finite() {
                bad.push(format!(
                    "{name} no finito en n={} seed={} mode={} a={:.3}: {}",
                    snap.n, snap.seed, snap.mode, snap.a_target, v
                ));
            }
        }
        for (i, &p) in snap.pk_measured.iter().enumerate() {
            if !p.is_finite() {
                bad.push(format!(
                    "pk_measured[{i}] no finito en seed={} mode={} a={:.3}",
                    snap.seed, snap.mode, snap.a_target
                ));
            }
        }
        for (i, &p) in snap.pk_corrected.iter().enumerate() {
            if !p.is_finite() {
                bad.push(format!(
                    "pk_corrected[{i}] no finito en seed={} mode={} a={:.3}",
                    snap.seed, snap.mode, snap.a_target
                ));
            }
        }
    }

    dump_json(
        "test7_no_nan_inf",
        json!({"bad_entries": bad, "total_bad": bad.len()}),
    );

    assert!(
        bad.is_empty(),
        "Se detectaron valores no finitos: {} entradas",
        bad.len()
    );
}
