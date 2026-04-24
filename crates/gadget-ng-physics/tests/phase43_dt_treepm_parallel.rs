//! Phase 43 — Control temporal de TreePM + paralelismo mínimo de loops calientes.
//!
//! Hipótesis (motivada por Phase 42):
//!
//! > Con `TreePM + ε_phys = 0.01 Mpc/h` el cuello dominante del error de
//! > crecimiento lineal ya **no** es el softening ni la resolución, sino el
//! > paso temporal. Si esto es cierto, un barrido de `dt` más fino debería
//! > mostrar mejora medible (contra Phase 39, que veía el mismo barrido
//! > sobre PM puro y no detectaba dependencia — el error estaba dominado
//! > por el softening nulo del corto alcance).
//!
//! ## Matriz (por default, smoke test)
//!
//! - `N = 32³`, `seed = 42`, `Z0Sigma8`, 2LPT, `a_init = 0.02`, snapshots
//!   en `a ∈ {0.02, 0.05, 0.10}`.
//! - Variantes **físicas** — todas TreePM con `ε_phys = 0.01 Mpc/h`:
//!     - `dt_4e-4` (default histórico Phase 42)
//!     - `dt_2e-4`
//!     - `dt_1e-4`
//!     - (opcional) `dt_5e-5` con `PHASE43_DT5E5=1`
//!     - `adaptive_cosmo`   — `dt = min(η·√(ε/a_max), κ_h·a/H(a))`
//!                            con `η = 0.1`, `κ_h = 0.04`,
//!                            `dt ∈ [5·10⁻⁵, 4·10⁻⁴]`
//! - `PHASE43_N=64` sube a `N = 64³` (coste ~8× respecto al smoke).
//! - `PHASE43_THREADS="1,4,8"` define el barrido de hilos Rayon para la
//!   medición de speedup (default `"1,4"`).
//! - `PHASE43_USE_CACHE=1` re-lee `target/phase43/per_snapshot_metrics.json`
//!   sin re-correr la matriz física.
//!
//! ## Tests (soft/hard coherente con Phase 41/42)
//!
//! 1. `treepm_softened_dt_sweep_runs_stably`              — hard
//! 2. `smaller_dt_improves_growth_under_treepm`           — soft
//! 3. `adaptive_dt_matches_or_beats_best_fixed_dt`        — soft
//! 4. `parallel_tree_walk_matches_serial_within_tolerance` — hard (tolerancia reproducibilidad)
//! 5. `parallel_execution_reduces_wall_time`              — soft
//! 6. `no_nan_inf_under_phase43_matrix`                   — hard
//! 7. `results_consistent_across_thread_counts`           — hard

use gadget_ng_analysis::pk_correction::{correct_pk, RnModel};
use gadget_ng_analysis::power_spectrum::{power_spectrum, PkBin};
use gadget_ng_core::{
    amplitude_for_sigma8, build_particles,
    cosmology::{gravity_coupling_qksl, growth_factor_d_ratio, CosmologyParams},
    transfer_eh_nowiggle, wrap_position, CosmologySection, EisensteinHuParams, GravitySection,
    GravitySolver, IcKind, InitialConditionsSection, NormalizationMode, OutputSection,
    PerformanceSection, RunConfig, SimulationSection, TimestepSection, TransferKind, UnitsSection,
    Vec3,
};
use gadget_ng_integrators::{
    compute_global_adaptive_dt, leapfrog_cosmo_kdk_step, AdaptiveDtCriterion, CosmoFactors,
};
use gadget_ng_treepm::TreePmSolver;
use serde_json::json;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

// ── Constantes físicas (coincidentes con Phase 41/42) ────────────────────────

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
const SEED: u64 = 42;

/// Softening físico **fijo** a Phase 42 óptimo (Mpc/h).
const EPS_PHYS_MPC_H: f64 = 0.01;

const N_DEFAULT: usize = 32;
const N_QUICK: usize = 32;

/// Base `dt` sweep (fijos) según el brief. El cuarto valor (`5e-5`) es
/// opcional y se activa con `PHASE43_DT5E5=1`.
const DT_SWEEP_BASE: [f64; 3] = [4.0e-4, 2.0e-4, 1.0e-4];
const DT_EXTRA_FINE: f64 = 5.0e-5;

/// Coeficiente adimensional del criterio adaptativo por aceleración
/// (`dt ≤ η·√(ε/a_max)`). Valor moderado: `0.1` da pasos ~`dt_stab/4`.
const ETA_ACCEL: f64 = 0.1;

/// Fracción del tiempo de Hubble que el timestep cosmológico permite:
/// `dt ≤ kappa_h · a / H(a)`. `0.04` → ≥25 pasos por e-folding de `a`.
const KAPPA_H: f64 = 0.04;

/// Cotas de `dt` para el modo adaptativo (en unidades del código).
/// `dt_min` = límite óptimo del barrido fijo (evita runaway en close-pairs
/// patológicas); `dt_max` = `dt` histórico de Phase 42 (no podemos ser
/// *más* gruesos).
const DT_MIN_ADAPTIVE: f64 = 5.0e-5;
const DT_MAX_ADAPTIVE: f64 = 4.0e-4;

// ── Variantes físicas del barrido ────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
enum DtMode {
    Fixed(f64),
    /// Criterio combinado `dt = min(η·√(ε/a_max), κ_h·a/H(a))`, clamped a
    /// `[DT_MIN_ADAPTIVE, DT_MAX_ADAPTIVE]`.
    AdaptiveCosmoAccel {
        eta: f64,
        kappa_h: f64,
    },
}

impl DtMode {
    fn label(&self) -> String {
        match self {
            DtMode::Fixed(dt) => format!("dt_{:.0e}", dt).replace("e0", "e"),
            DtMode::AdaptiveCosmoAccel { .. } => "adaptive_cosmo".into(),
        }
    }
}

fn dt_variants() -> Vec<DtMode> {
    // `PHASE43_QUICK=1` recorta el sweep a {4e-4, 2e-4} (smoke < 10 min a N=32).
    let sweep: Vec<f64> = if env_flag("PHASE43_QUICK") {
        vec![DT_SWEEP_BASE[0], DT_SWEEP_BASE[1]]
    } else {
        DT_SWEEP_BASE.to_vec()
    };
    let mut v: Vec<DtMode> = sweep.iter().copied().map(DtMode::Fixed).collect();
    if env_flag("PHASE43_DT5E5") {
        v.push(DtMode::Fixed(DT_EXTRA_FINE));
    }
    if !env_flag("PHASE43_SKIP_ADAPTIVE") {
        v.push(DtMode::AdaptiveCosmoAccel {
            eta: ETA_ACCEL,
            kappa_h: KAPPA_H,
        });
    }
    v
}

// ── Utilidades de entorno ────────────────────────────────────────────────────

fn env_flag(name: &str) -> bool {
    std::env::var(name).map(|v| v == "1").unwrap_or(false)
}

fn n_grid() -> usize {
    if let Ok(v) = std::env::var("PHASE43_N") {
        if let Ok(n) = v.parse::<usize>() {
            if n.is_power_of_two() && (16..=256).contains(&n) {
                return n;
            }
        }
    }
    if env_flag("PHASE43_QUICK") {
        N_QUICK
    } else {
        N_DEFAULT
    }
}

fn thread_counts() -> Vec<usize> {
    let raw = std::env::var("PHASE43_THREADS").unwrap_or_else(|_| "1,4".to_string());
    let mut out: Vec<usize> = raw
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1 && n <= 128)
        .collect();
    if out.is_empty() {
        out = vec![1, 4];
    }
    out
}

// ── Helpers físicos (clones mínimos de Phase 42 para desacoplar) ─────────────

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

fn build_run_config(n: usize, seed: u64) -> RunConfig {
    let gravity = GravitySection {
        solver: gadget_ng_core::SolverKind::TreePm,
        pm_grid_size: n,
        ..GravitySection::default()
    };
    RunConfig {
        simulation: SimulationSection {
            dt: DT_SWEEP_BASE[0],
            num_steps: 10,
            softening: EPS_PHYS_MPC_H / BOX_MPC_H,
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
                normalization_mode: NormalizationMode::Z0Sigma8,
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
        rt: Default::default(), reionization: Default::default(), mhd: Default::default(),
    }
}

fn measure_pk(parts: &[gadget_ng_core::Particle], n: usize) -> Vec<PkBin> {
    let positions: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    power_spectrum(&positions, &masses, BOX, n)
}

/// Una evaluación de aceleraciones TreePM con `ε_phys = EPS_PHYS_MPC_H`.
fn compute_treepm_accels(
    parts: &[gadget_ng_core::Particle],
    n_mesh: usize,
    g_cosmo: f64,
    out: &mut [Vec3],
) {
    let n_p = parts.len();
    let positions: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    let indices: Vec<usize> = (0..n_p).collect();

    let eps = EPS_PHYS_MPC_H / BOX_MPC_H;
    let eps2 = eps * eps;
    let tpm = TreePmSolver {
        grid_size: n_mesh,
        box_size: BOX,
        r_split: 0.0,
    };
    tpm.accelerations_for_indices(&positions, &masses, eps2, g_cosmo, &indices, out);
}

/// Evoluciona `parts` desde `a_start` a `a_target` con `dt_mode`. Devuelve
/// `(a_final, n_steps, dt_trace)`. `dt_trace` contiene el `dt` efectivo
/// usado en cada paso (para el modo adaptativo).
fn evolve_with_dt_mode(
    parts: &mut Vec<gadget_ng_core::Particle>,
    n_mesh: usize,
    a_start: f64,
    a_target: f64,
    dt_mode: DtMode,
) -> (f64, usize, Vec<(f64, f64)>) {
    if a_start >= a_target {
        return (a_start, 0, Vec::new());
    }
    let cosmo = cosmo_params();
    let mut scratch = vec![Vec3::zero(); parts.len()];
    let mut a = a_start;
    let mut dt_trace: Vec<(f64, f64)> = Vec::new();
    let mut steps = 0usize;
    let max_iter = 1_000_000;

    // Para el modo adaptativo necesitamos una evaluación inicial de aceleraciones
    // que fije el primer `dt`. Lo hacemos con el default de `DT_SWEEP_BASE[0]`
    // como placeholder: el verdadero `dt` se determina a partir del campo de
    // aceleraciones actualizado dentro del bucle.
    let eps_internal = EPS_PHYS_MPC_H / BOX_MPC_H;

    for _ in 0..max_iter {
        if a >= a_target {
            break;
        }

        // ── Elegir `dt` para este paso ─────────────────────────────────────────
        let dt = match dt_mode {
            DtMode::Fixed(dtf) => dtf,
            DtMode::AdaptiveCosmoAccel { eta, kappa_h } => {
                // Evaluar aceleraciones actuales y derivar dt.
                let g_cosmo = gravity_coupling_qksl(G, a);
                compute_treepm_accels(parts, n_mesh, g_cosmo, &mut scratch);
                let crit = AdaptiveDtCriterion::cosmo_acceleration(
                    eta,
                    eps_internal,
                    kappa_h,
                    DT_MIN_ADAPTIVE,
                    DT_MAX_ADAPTIVE,
                );
                compute_global_adaptive_dt(crit, &scratch, Some(cosmo), a)
            }
        };

        // Recortar último sub-paso para aterrizar exactamente en `a_target`
        // (pseudo-retrocompatibilidad con Phase 42 que evolucionaba "pasado")
        let dt_used = dt;

        let g_cosmo = gravity_coupling_qksl(G, a);
        let (drift, kick_half, kick_half2) = cosmo.drift_kick_factors(a, dt_used);
        let cf = CosmoFactors {
            drift,
            kick_half,
            kick_half2,
        };
        let a_prev = a;
        a = cosmo.advance_a(a, dt_used);
        leapfrog_cosmo_kdk_step(parts, cf, &mut scratch, |ps, acc| {
            compute_treepm_accels(ps, n_mesh, g_cosmo, acc);
        });
        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
        dt_trace.push((a_prev, dt_used));
        steps += 1;
    }
    (a, steps, dt_trace)
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

fn p_ref_at_a(k_hmpc: f64, a: f64) -> f64 {
    let cosmo = cosmo_params();
    let d_ratio = growth_factor_d_ratio(cosmo, a, 1.0);
    eh_pk_at_z0(k_hmpc) * d_ratio * d_ratio
}

#[inline]
fn shot_noise_level(n_grid: usize) -> f64 {
    let vol = BOX_MPC_H.powi(3);
    let np = (n_grid as f64).powi(3);
    vol / np
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
    let sum_v2: f64 = parts
        .iter()
        .map(|p| {
            let v = p.velocity;
            v.x * v.x + v.y * v.y + v.z * v.z
        })
        .sum();
    (sum_v2 / parts.len() as f64).sqrt()
}

// ── Estructuras de resultados ────────────────────────────────────────────────

#[derive(Clone)]
struct SnapshotResult {
    n: usize,
    seed: u64,
    variant: String,
    dt_nominal: f64,
    is_adaptive: bool,
    a_target: f64,
    a_actual: f64,
    n_steps: usize,
    wall_time_s: f64,
    ks_hmpc: Vec<f64>,
    pk_corrected: Vec<f64>,
    pk_reference: Vec<f64>,
    p_shot: f64,
    delta_rms: f64,
    v_rms: f64,
    dt_trace: Vec<(f64, f64)>,
}

impl SnapshotResult {
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

    fn from_json_value(v: &serde_json::Value) -> Option<Self> {
        let ks: Vec<f64> = v
            .get("ks_hmpc")?
            .as_array()?
            .iter()
            .filter_map(|x| x.as_f64())
            .collect();
        let pc: Vec<f64> = v
            .get("pk_corrected_mpc_h3")?
            .as_array()?
            .iter()
            .filter_map(|x| x.as_f64())
            .collect();
        let pr: Vec<f64> = v
            .get("pk_reference_mpc_h3")?
            .as_array()?
            .iter()
            .filter_map(|x| x.as_f64())
            .collect();
        let trace: Vec<(f64, f64)> = v
            .get("dt_trace")
            .and_then(|x| x.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|row| {
                        let r = row.as_array()?;
                        Some((r.get(0)?.as_f64()?, r.get(1)?.as_f64()?))
                    })
                    .collect()
            })
            .unwrap_or_default();
        Some(Self {
            n: v.get("n")?.as_u64()? as usize,
            seed: v.get("seed")?.as_u64()?,
            variant: v.get("variant")?.as_str()?.to_string(),
            dt_nominal: v.get("dt_nominal")?.as_f64()?,
            is_adaptive: v.get("is_adaptive")?.as_bool().unwrap_or(false),
            a_target: v.get("a_target")?.as_f64()?,
            a_actual: v.get("a_actual")?.as_f64()?,
            n_steps: v.get("n_steps")?.as_u64().unwrap_or(0) as usize,
            wall_time_s: v.get("wall_time_s")?.as_f64().unwrap_or(f64::NAN),
            ks_hmpc: ks,
            pk_corrected: pc,
            pk_reference: pr,
            p_shot: v.get("p_shot_mpc_h3")?.as_f64()?,
            delta_rms: v.get("delta_rms")?.as_f64()?,
            v_rms: v.get("v_rms")?.as_f64()?,
            dt_trace: trace,
        })
    }

    fn to_json(&self) -> serde_json::Value {
        let r = self.r_corr();
        json!({
            "n": self.n,
            "seed": self.seed,
            "variant": self.variant,
            "dt_nominal": self.dt_nominal,
            "is_adaptive": self.is_adaptive,
            "a_target": self.a_target,
            "a_actual": self.a_actual,
            "n_steps": self.n_steps,
            "wall_time_s": self.wall_time_s,
            "ks_hmpc": self.ks_hmpc,
            "pk_corrected_mpc_h3": self.pk_corrected,
            "pk_reference_mpc_h3": self.pk_reference,
            "p_shot_mpc_h3": self.p_shot,
            "median_abs_log10_err_corrected": median_abs(&self.log_err_corr()),
            "mean_r_corr": mean(&r),
            "std_r_corr": stdev(&r),
            "cv_r_corr": stdev(&r) / mean(&r).abs(),
            "delta_rms": self.delta_rms,
            "v_rms": self.v_rms,
            "dt_trace": self.dt_trace.iter().map(|(a,d)| json!([a,d])).collect::<Vec<_>>(),
        })
    }
}

// ── Cálculo de métricas de crecimiento ───────────────────────────────────────

fn find_opt<'a>(
    m: &'a [SnapshotResult],
    variant: &str,
    a_target: f64,
) -> Option<&'a SnapshotResult> {
    m.iter()
        .find(|r| r.variant == variant && (r.a_target - a_target).abs() < 1e-9)
}

fn growth_ratio_low_k(
    m: &[SnapshotResult],
    variant: &str,
    a_target: f64,
    k_max_hmpc: f64,
) -> Option<(f64, f64, f64)> {
    let ic = find_opt(m, variant, A_INIT)?;
    let ev = find_opt(m, variant, a_target)?;

    let mut ratios = Vec::new();
    for (i, &k) in ev.ks_hmpc.iter().enumerate() {
        if k > k_max_hmpc {
            break;
        }
        if let Some(j) = ic.ks_hmpc.iter().position(|&k_ic| (k_ic - k).abs() < 1e-9) {
            let p_ev = ev.pk_corrected[i];
            let p_ic = ic.pk_corrected[j];
            if p_ic > 0.0 && p_ev > 0.0 {
                ratios.push(p_ev / p_ic);
            }
        }
    }
    if ratios.is_empty() {
        return None;
    }
    let measured = mean(&ratios);
    let cosmo = cosmo_params();
    let d_ratio = growth_factor_d_ratio(cosmo, a_target, A_INIT);
    let theory = d_ratio * d_ratio;
    let rel_err = (measured - theory).abs() / theory;
    Some((measured, theory, rel_err))
}

// ── Orquestación física ──────────────────────────────────────────────────────

fn run_one_variant(n: usize, seed: u64, mode: DtMode) -> Vec<SnapshotResult> {
    let cfg = build_run_config(n, seed);
    let mut parts = build_particles(&cfg).expect("build_particles falló");
    let model = RnModel::phase35_default();
    let p_shot = shot_noise_level(n);

    let mut results = Vec::with_capacity(A_SNAPSHOTS.len());
    let mut a_current = A_INIT;
    let label = mode.label();
    let (dt_nominal, is_adaptive) = match mode {
        DtMode::Fixed(dt) => (dt, false),
        DtMode::AdaptiveCosmoAccel { .. } => (f64::NAN, true),
    };
    let mut total_trace: Vec<(f64, f64)> = Vec::new();
    let mut total_steps = 0usize;

    for &a_t in A_SNAPSHOTS.iter() {
        let t0 = Instant::now();
        if a_current < a_t {
            let (a_next, n_steps, trace) = evolve_with_dt_mode(&mut parts, n, a_current, a_t, mode);
            a_current = a_next;
            total_steps += n_steps;
            total_trace.extend(trace);
        }
        let wall = t0.elapsed().as_secs_f64();
        let pk_raw = measure_pk(&parts, n);
        let pk_win = linear_window(&pk_raw, n);
        let pk_corr = correct_pk(&pk_win, BOX, n, None, &model);

        let mut ks = Vec::new();
        let mut pc_v = Vec::new();
        let mut pr_v = Vec::new();
        for (bin_m, bin_c) in pk_win.iter().zip(pk_corr.iter()) {
            let k_h = k_internal_to_hmpc(bin_m.k);
            let pref = p_ref_at_a(k_h, a_current);
            if bin_m.pk > 0.0 && bin_c.pk > 0.0 && pref > 0.0 && pref.is_finite() {
                ks.push(k_h);
                pc_v.push(bin_c.pk);
                pr_v.push(pref);
            }
        }

        let d_rms = delta_rms(&parts, n);
        let vrms = v_rms(&parts);
        let err_med = median_abs(
            &pc_v
                .iter()
                .zip(pr_v.iter())
                .map(|(&c, &r)| (c / r).log10())
                .collect::<Vec<_>>(),
        );
        eprintln!(
            "[phase43] N={n:<4} var={label:<16} a={a_t:.2}→{a_current:.3}  steps={total_steps:<5} \
             δ_rms={d_rms:.3e}  v_rms={vrms:.3e}  err={err_med:.3e}  wall={wall:.1}s"
        );

        results.push(SnapshotResult {
            n,
            seed,
            variant: label.clone(),
            dt_nominal,
            is_adaptive,
            a_target: a_t,
            a_actual: a_current,
            n_steps: total_steps,
            wall_time_s: wall,
            ks_hmpc: ks,
            pk_corrected: pc_v,
            pk_reference: pr_v,
            p_shot,
            delta_rms: d_rms,
            v_rms: vrms,
            dt_trace: total_trace.clone(),
        });
    }
    results
}

fn run_full_matrix() -> Vec<SnapshotResult> {
    let n = n_grid();
    let mut all = Vec::new();
    for mode in dt_variants() {
        let t0 = Instant::now();
        let sims = run_one_variant(n, SEED, mode);
        let el = t0.elapsed().as_secs_f64();
        eprintln!(
            "[phase43] ✓ N={n} var={:<16} completada en {el:.1}s",
            mode.label()
        );
        all.extend(sims);
    }
    all
}

fn matrix() -> &'static [SnapshotResult] {
    static CELL: OnceLock<Vec<SnapshotResult>> = OnceLock::new();
    CELL.get_or_init(|| {
        if env_flag("PHASE43_USE_CACHE") {
            let mut path = phase43_dir();
            path.push("per_snapshot_metrics.json");
            if path.exists() {
                if let Ok(txt) = fs::read_to_string(&path) {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&txt) {
                        if let Some(arr) = val.get("snapshots").and_then(|v| v.as_array()) {
                            eprintln!(
                                "[phase43] cargando matriz cacheada ({} snapshots) de {}",
                                arr.len(),
                                path.display()
                            );
                            let out: Vec<SnapshotResult> = arr
                                .iter()
                                .filter_map(SnapshotResult::from_json_value)
                                .collect();
                            if !out.is_empty() {
                                return out;
                            }
                        }
                    }
                }
            }
        }
        run_full_matrix()
    })
}

fn phase43_dir() -> PathBuf {
    let mut d = PathBuf::from(std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.pop();
        p.push("target");
        p.to_string_lossy().to_string()
    }));
    d.push("phase43");
    let _ = fs::create_dir_all(&d);
    d
}

fn dump_json(name: &str, value: serde_json::Value) {
    let mut path = phase43_dir();
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
    let labels: Vec<String> = dt_variants().iter().map(|v| v.label()).collect();
    dump_json(
        "per_snapshot_metrics",
        json!({
            "n": n_grid(),
            "seed": SEED,
            "a_snapshots": A_SNAPSHOTS,
            "box_mpc_h": BOX_MPC_H,
            "eps_phys_mpc_h": EPS_PHYS_MPC_H,
            "eta_accel": ETA_ACCEL,
            "dt_base_sweep": DT_SWEEP_BASE,
            "variants": labels,
            "snapshots": all,
        }),
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 1 — treepm_softened_dt_sweep_runs_stably                      HARD
// ══════════════════════════════════════════════════════════════════════════════
//
// Verifica que cada corrida del barrido llegó a `a ≈ 0.10` (tolerancia dt)
// y produjo snapshots no degenerados.

#[test]
fn treepm_softened_dt_sweep_runs_stably() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let mut bad = Vec::new();
    for mode in dt_variants() {
        let label = mode.label();
        for &a_t in A_SNAPSHOTS.iter() {
            let Some(s) = find_opt(m, &label, a_t) else {
                bad.push(format!("snapshot ausente: var={label} a={a_t:.3}"));
                continue;
            };
            // Tolerancia para el sobrepaso en el último sub-paso
            if s.a_actual < a_t - 1e-9 || s.a_actual > a_t + 0.02 {
                bad.push(format!(
                    "a_actual fuera de rango en var={label} a_t={a_t:.3}: a={:.4}",
                    s.a_actual
                ));
            }
            if s.pk_corrected.is_empty() {
                bad.push(format!("pk_corrected vacío en var={label} a_t={a_t:.3}"));
            }
        }
    }
    dump_json(
        "test1_dt_sweep_stable",
        json!({"bad_entries": bad, "total_bad": bad.len()}),
    );
    assert!(bad.is_empty(), "Phase 43 instabilidades: {:?}", bad);
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 2 — smaller_dt_improves_growth_under_treepm                   SOFT
// ══════════════════════════════════════════════════════════════════════════════
//
// Reporta si, dentro del barrido fijo, reducir `dt` mejora monótona
// o parcialmente el error de crecimiento a `a=0.10` en `k ≤ 0.1 h/Mpc`.
// Decisión A/B/C:
//   - A_smaller_dt_improves_growth      : error(dt=1e-4) < error(dt=4e-4)
//                                          por al menos 5 %.
//   - B_dt_has_no_measurable_effect     : error prácticamente plano.
//   - C_smaller_dt_worsens_growth       : el error aumenta al reducir dt.

#[test]
fn smaller_dt_improves_growth_under_treepm() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let k_max = 0.1;
    let a_t = 0.10;

    let mut curve: Vec<(f64, f64, String)> = Vec::new();
    for mode in dt_variants() {
        let DtMode::Fixed(dt) = mode else { continue };
        let label = mode.label();
        if let Some((_, _, rel)) = growth_ratio_low_k(m, &label, a_t, k_max) {
            curve.push((dt, rel, label));
        }
    }
    curve.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // dt descendente: 4e-4 → 5e-5

    let n_pts = curve.len();
    let first_err = curve.first().map(|x| x.1).unwrap_or(f64::NAN);
    let last_err = curve.last().map(|x| x.1).unwrap_or(f64::NAN);

    let threshold = 0.05; // 5 % de mejora relativa
    let decision = if n_pts < 2 {
        "D_insufficient_points"
    } else {
        let rel_change = (first_err - last_err) / first_err.max(f64::EPSILON);
        if rel_change > threshold {
            "A_smaller_dt_improves_growth"
        } else if rel_change < -threshold {
            "C_smaller_dt_worsens_growth"
        } else {
            "B_dt_has_no_measurable_effect"
        }
    };

    let curve_json: Vec<_> = curve
        .iter()
        .map(|(dt, err, lbl)| {
            json!({
                "variant": lbl,
                "dt": dt,
                "rel_err_growth_a010": err,
            })
        })
        .collect();

    dump_json(
        "test2_smaller_dt_improves_growth",
        json!({
            "curve": curve_json,
            "decision": decision,
            "first_err": first_err,
            "last_err": last_err,
            "threshold": threshold,
            "a_target": a_t,
            "k_max_hmpc": k_max,
        }),
    );

    eprintln!(
        "[phase43][test2] DECISION={decision}  first_err={first_err:.3e}  last_err={last_err:.3e}"
    );

    for (_, e, l) in &curve {
        assert!(e.is_finite(), "rel_err no finito en {l}: {e}");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 3 — adaptive_dt_matches_or_beats_best_fixed_dt                SOFT
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn adaptive_dt_matches_or_beats_best_fixed_dt() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let k_max = 0.1;
    let a_t = 0.10;

    // Mejor dt fijo
    let mut best_fixed: Option<(String, f64, f64)> = None; // (label, dt, err)
    for mode in dt_variants() {
        let DtMode::Fixed(dt) = mode else { continue };
        let label = mode.label();
        if let Some((_, _, err)) = growth_ratio_low_k(m, &label, a_t, k_max) {
            if best_fixed
                .as_ref()
                .map(|(_, _, e)| err < *e)
                .unwrap_or(true)
            {
                best_fixed = Some((label, dt, err));
            }
        }
    }

    // Adaptativo
    let adapt_label = "adaptive_cosmo".to_string();
    let adapt_err = growth_ratio_low_k(m, &adapt_label, a_t, k_max).map(|(_, _, e)| e);

    // Steps + wall time adaptativo vs best_fixed
    let adapt_snap = find_opt(m, &adapt_label, a_t);
    let fixed_snap = best_fixed
        .as_ref()
        .and_then(|(l, _, _)| find_opt(m, l, a_t));

    let decision = match (adapt_err, best_fixed.as_ref()) {
        (Some(ae), Some((_, _, be))) => {
            if ae < *be * 0.95 {
                "A_adaptive_beats_best_fixed"
            } else if ae <= *be * 1.05 {
                "B_adaptive_matches_best_fixed"
            } else {
                "C_adaptive_worse_than_best_fixed"
            }
        }
        (None, _) => "D_adaptive_not_available",
        _ => "D_no_fixed_baseline",
    };

    dump_json(
        "test3_adaptive_vs_fixed",
        json!({
            "best_fixed_variant": best_fixed.as_ref().map(|(l,_,_)| l.clone()),
            "best_fixed_dt": best_fixed.as_ref().map(|(_,d,_)| *d),
            "best_fixed_err": best_fixed.as_ref().map(|(_,_,e)| *e),
            "adaptive_err": adapt_err,
            "adaptive_steps": adapt_snap.map(|s| s.n_steps),
            "adaptive_wall_s": adapt_snap.map(|s| s.wall_time_s),
            "fixed_steps": fixed_snap.map(|s| s.n_steps),
            "fixed_wall_s": fixed_snap.map(|s| s.wall_time_s),
            "decision": decision,
            "a_target": a_t,
            "k_max_hmpc": k_max,
        }),
    );

    eprintln!(
        "[phase43][test3] DECISION={decision}  adapt_err={:?}  best_fixed_err={:?}",
        adapt_err,
        best_fixed.as_ref().map(|(_, _, e)| *e)
    );

    if let Some(e) = adapt_err {
        assert!(e.is_finite(), "adaptive rel_err no finito: {e}");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 4 — parallel_tree_walk_matches_serial_within_tolerance        HARD
// ══════════════════════════════════════════════════════════════════════════════
//
// Para un único step TreePM sobre las ICs (sin integrar), compara las
// aceleraciones calculadas bajo pools de 1 vs 4 hilos. La tolerancia relaja
// el cero absoluto por la suma no asociativa en punto flotante, pero es
// estricta en precisión relativa.

#[test]
fn parallel_tree_walk_matches_serial_within_tolerance() {
    let n = n_grid();
    let cfg = build_run_config(n, SEED);
    let parts = build_particles(&cfg).expect("build_particles falló");
    let g_cosmo = G / A_INIT;

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pool4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    let mut acc1 = vec![Vec3::zero(); parts.len()];
    let mut acc4 = vec![Vec3::zero(); parts.len()];

    pool1.install(|| compute_treepm_accels(&parts, n, g_cosmo, &mut acc1));
    pool4.install(|| compute_treepm_accels(&parts, n, g_cosmo, &mut acc4));

    // Escala de aceleración característica (mediana de |a1|) para normalizar.
    let mut mags: Vec<f64> = acc1
        .iter()
        .map(|a| (a.x * a.x + a.y * a.y + a.z * a.z).sqrt())
        .filter(|v| v.is_finite())
        .collect();
    mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = if mags.is_empty() {
        0.0
    } else {
        mags[mags.len() / 2]
    };

    let mut max_abs_diff = 0.0_f64;
    let mut max_rel_diff = 0.0_f64;
    for (a, b) in acc1.iter().zip(acc4.iter()) {
        let dx = (a.x - b.x).abs();
        let dy = (a.y - b.y).abs();
        let dz = (a.z - b.z).abs();
        let diff = dx.max(dy).max(dz);
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
        let scale = (a.x * a.x + a.y * a.y + a.z * a.z).sqrt().max(med * 1e-3);
        let rel = diff / scale.max(1e-30);
        if rel > max_rel_diff {
            max_rel_diff = rel;
        }
    }

    // Tolerancia: la suma paralela difiere de la serial a nivel de bits por
    // orden de reducción; en doble precisión el ruido típico es 1e-12 relativo.
    let rel_tol = 1e-10;
    dump_json(
        "test4_parallel_matches_serial",
        json!({
            "n": n,
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
            "rel_tolerance": rel_tol,
            "median_accel_magnitude": med,
        }),
    );
    assert!(
        max_rel_diff < rel_tol,
        "Paralelismo TreePM difiere de la serial por más de {rel_tol}: max_rel={max_rel_diff:.3e}"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 5 — parallel_execution_reduces_wall_time                      SOFT
// ══════════════════════════════════════════════════════════════════════════════
//
// Mide el wall-time de **un step TreePM** bajo cada número de hilos en
// `PHASE43_THREADS`, relativo a 1 hilo. El test sólo reporta y falla si el
// speedup con `≥ 4` hilos no supera `1.1×`.

#[test]
fn parallel_execution_reduces_wall_time() {
    let n = n_grid();
    let cfg = build_run_config(n, SEED);
    let parts = build_particles(&cfg).expect("build_particles falló");
    let g_cosmo = G / A_INIT;

    let counts = thread_counts();
    let mut rows: Vec<(usize, f64)> = Vec::new();
    for &t in &counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap();
        let mut acc = vec![Vec3::zero(); parts.len()];
        // Warmup (eviction de caches, llenado de thread pool)
        pool.install(|| compute_treepm_accels(&parts, n, g_cosmo, &mut acc));
        let t0 = Instant::now();
        pool.install(|| compute_treepm_accels(&parts, n, g_cosmo, &mut acc));
        let el = t0.elapsed().as_secs_f64();
        rows.push((t, el));
        eprintln!("[phase43][test5] threads={t:<3} wall={el:.3}s");
    }

    let t1 = rows.iter().find(|(t, _)| *t == 1).map(|(_, v)| *v);
    let decision_rows: Vec<_> = rows
        .iter()
        .map(|(t, v)| {
            let su = t1.map(|b| b / *v).unwrap_or(f64::NAN);
            json!({"threads": t, "wall_s": v, "speedup_vs_1": su})
        })
        .collect();

    let speedup_at_highest = rows
        .iter()
        .max_by_key(|(t, _)| *t)
        .and_then(|(_, v)| t1.map(|b| b / *v));

    let decision = match speedup_at_highest {
        Some(su) if su >= 1.5 => "A_clear_parallel_speedup",
        Some(su) if su >= 1.1 => "B_mild_parallel_speedup",
        Some(_) => "C_parallel_no_speedup",
        None => "D_insufficient_points",
    };

    dump_json(
        "test5_parallel_speedup",
        json!({
            "n": n,
            "rows": decision_rows,
            "decision": decision,
            "speedup_at_highest": speedup_at_highest,
        }),
    );

    // Hard check: rows should be finite positive.
    for (_, v) in &rows {
        assert!(v.is_finite() && *v > 0.0, "wall-time no válido: {v}");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 6 — no_nan_inf_under_phase43_matrix                           HARD
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn no_nan_inf_under_phase43_matrix() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let mut bad = Vec::new();
    for snap in m {
        let metrics = [
            ("p_shot", snap.p_shot),
            ("delta_rms", snap.delta_rms),
            ("v_rms", snap.v_rms),
            ("a_actual", snap.a_actual),
        ];
        for (name, v) in metrics.iter() {
            if !v.is_finite() {
                bad.push(format!(
                    "{name} no finito en var={} a={:.3}: {}",
                    snap.variant, snap.a_target, v
                ));
            }
        }
        for (i, &p) in snap.pk_corrected.iter().enumerate() {
            if !p.is_finite() {
                bad.push(format!(
                    "pk_corrected[{i}] no finito en var={} a={:.3}",
                    snap.variant, snap.a_target
                ));
            }
        }
    }
    dump_json(
        "test6_no_nan_inf",
        json!({"bad_entries": bad, "total_bad": bad.len()}),
    );
    assert!(
        bad.is_empty(),
        "Phase 43 detectó {} valores no finitos",
        bad.len()
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 7 — results_consistent_across_thread_counts                   HARD
// ══════════════════════════════════════════════════════════════════════════════
//
// Comprueba que una corrida **completa** (drift+kick hasta `a=0.05`) con
// 1 vs 4 hilos produce `δ_rms(0.05)` y `v_rms(0.05)` dentro de tolerancia
// numérica relativa.

#[test]
fn results_consistent_across_thread_counts() {
    let n = n_grid();
    let run_one = |threads: usize| -> (f64, f64, f64) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let cfg = build_run_config(n, SEED);
        let mut parts = build_particles(&cfg).expect("build_particles falló");
        let mut a = A_INIT;
        pool.install(|| {
            let (a_next, _, _) =
                evolve_with_dt_mode(&mut parts, n, a, 0.05, DtMode::Fixed(DT_SWEEP_BASE[0]));
            a = a_next;
        });
        let d = delta_rms(&parts, n);
        let v = v_rms(&parts);
        (a, d, v)
    };

    let (a1, d1, v1) = run_one(1);
    let (a4, d4, v4) = run_one(4);

    let tol = 1e-6;
    let da = (a1 - a4).abs() / a1.abs().max(1e-30);
    let dd = (d1 - d4).abs() / d1.abs().max(1e-30);
    let dv = (v1 - v4).abs() / v1.abs().max(1e-30);

    dump_json(
        "test7_parallel_consistency",
        json!({
            "n": n,
            "a_rel_diff": da,
            "delta_rms_rel_diff": dd,
            "v_rms_rel_diff": dv,
            "tolerance": tol,
            "ts1": {"a": a1, "delta_rms": d1, "v_rms": v1},
            "ts4": {"a": a4, "delta_rms": d4, "v_rms": v4},
        }),
    );

    assert!(
        da < tol,
        "a inconsistente entre 1 y 4 hilos: rel_diff={da:.3e}"
    );
    assert!(
        dd < 1e-3,
        "delta_rms inconsistente entre 1 y 4 hilos: rel_diff={dd:.3e}"
    );
    assert!(
        dv < 1e-3,
        "v_rms inconsistente entre 1 y 4 hilos: rel_diff={dv:.3e}"
    );
}
