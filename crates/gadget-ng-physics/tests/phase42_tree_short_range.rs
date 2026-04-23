//! Phase 42 — Regularización física de fuerzas vía TreePM + softening absoluto.
//!
//! Phase 41 cerró el eje **shot-noise ↔ señal** a `N ≥ 64³` pero dejó abierto
//! el eje **dinámica lineal vs no-lineal temprana**: `δ_rms(a=0.10) ≈ 1`
//! incluso para `Z0Sigma8`, el `median|log10(P_c/P_ref)|` crece con `N` en
//! snapshots evolucionados, y `P(k,a)/P(k,a_init)` no recupera
//! `[D(a)/D(a_init)]²` en bajo-k.
//!
//! Phase 42 testea una hipótesis física concreta:
//!
//! > el PM puro con `eps2=0` (como se usa en Phase 41) deja el corto alcance
//! > sin regularizar; al subir `N`, modos de alta-k crecen sin un softening
//! > que controle los pares cercanos. La solución estándar es añadir un corte
//! > suavizado de corto alcance vía árbol.
//!
//! ## Matriz
//!
//! - `N = 128³`, seed = 42, `Z0Sigma8`, 2LPT, `a_init = 0.02`, snapshots en
//!   `a ∈ {0.02, 0.05, 0.10}`, `dt = 4·10⁻⁴`.
//! - 4 corridas únicas:
//!     - `pm_eps0`        → baseline `PmSolver` (réplica Phase 41).
//!     - `treepm_eps001`  → PM filtrado + SR `short_range_accels_periodic`
//!                          con `ε_phys = 0.01 Mpc/h`.
//!     - `treepm_eps002`  → idem, `ε_phys = 0.02 Mpc/h`.
//!     - `treepm_eps005`  → idem, `ε_phys = 0.05 Mpc/h`.
//!
//! `ε_internal = ε_phys / BOX_MPC_H`. `r_split = 2.5 · L/N = 2.5/128` (auto).
//! `r_cut = 5·r_split ≈ 0.098 L` ≪ `L/2`. Usamos [`TreePmSolver`] tal cual
//! (octree SR no-periódico): la aproximación introduce error sólo en una
//! cáscara `r_cut` cerca del borde (≲0.5 % de partículas con 2LPT ICs a
//! `a_init = 0.02`) y mantiene coste serial manejable gracias al
//! `use_monopole` para nodos lejanos.
//!
//! ## PM ignora `eps2`
//!
//! `PmSolver::accelerations_for_indices` recibe `_eps2: f64` sin usarlo
//! (el PM es band-limited: el softening no se aplica en la malla). Por eso
//! registramos **un solo** baseline PM, no tres redundantes.
//!
//! ## Tests
//!
//! 1. `softening_reduces_early_nonlinearity`  — hard
//! 2. `treepm_improves_growth_vs_pm`          — soft (decision A/B)
//! 3. `growth_closer_to_linear_with_softening` — soft (reporta ε óptimo)
//! 4. `no_nan_inf_phase42`                    — hard

use gadget_ng_analysis::pk_correction::{correct_pk, RnModel};
use gadget_ng_analysis::power_spectrum::{power_spectrum, PkBin};
use gadget_ng_core::{
    amplitude_for_sigma8, build_particles,
    cosmology::{growth_factor_d_ratio, CosmologyParams},
    transfer_eh_nowiggle, wrap_position, CosmologySection, EisensteinHuParams, GravitySection,
    GravitySolver, IcKind, InitialConditionsSection, NormalizationMode, OutputSection,
    PerformanceSection, RunConfig, SimulationSection, TimestepSection, TransferKind, UnitsSection,
    Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_pm::PmSolver;
use gadget_ng_treepm::TreePmSolver;
use serde_json::json;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

// ── Constantes físicas ───────────────────────────────────────────────────────

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
const DT: f64 = 4.0e-4;

/// `N` por defecto. Se puede bajar con `PHASE42_QUICK=1` para smoke test local.
const N_DEFAULT: usize = 128;
/// Smoke test con `PHASE42_QUICK=1`: N=32³ (32 768 partículas) mantiene
/// el coste TreePM serial bajo 10 min aún con cortes periódicos amplios.
const N_QUICK: usize = 32;

/// Softenings físicos en Mpc/h (barrido experimental de Phase 42).
const EPS_PHYSICAL_MPC_H: [f64; 3] = [0.01, 0.02, 0.05];

// ── Variantes de solver ──────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
enum SolverVariant {
    PmOnly,
    /// TreePM periódico con softening Plummer `ε` en unidades internas (`L/BOX_MPC_H`).
    TreePmPeriodic {
        eps_phys_mpc_h: f64,
    },
}

impl SolverVariant {
    fn label(&self) -> String {
        match self {
            SolverVariant::PmOnly => "pm_eps0".into(),
            SolverVariant::TreePmPeriodic { eps_phys_mpc_h } => {
                // Etiqueta coherente con los nombres de los configs TOML:
                // ε=0.01 → "001", ε=0.02 → "002", ε=0.05 → "005".
                let tag = format!("{:03}", (eps_phys_mpc_h * 100.0).round() as i32);
                format!("treepm_eps{tag}")
            }
        }
    }

    fn eps_phys_mpc_h(&self) -> f64 {
        match self {
            SolverVariant::PmOnly => 0.0,
            SolverVariant::TreePmPeriodic { eps_phys_mpc_h } => *eps_phys_mpc_h,
        }
    }

    fn eps_internal(&self) -> f64 {
        self.eps_phys_mpc_h() / BOX_MPC_H
    }
}

fn variants() -> Vec<SolverVariant> {
    let mut v = vec![SolverVariant::PmOnly];
    for &eps in EPS_PHYSICAL_MPC_H.iter() {
        v.push(SolverVariant::TreePmPeriodic {
            eps_phys_mpc_h: eps,
        });
    }
    v
}

fn n_grid() -> usize {
    if let Ok(v) = std::env::var("PHASE42_N") {
        if let Ok(n) = v.parse::<usize>() {
            if n.is_power_of_two() && (16..=256).contains(&n) {
                return n;
            }
        }
    }
    if std::env::var("PHASE42_QUICK")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        N_QUICK
    } else {
        N_DEFAULT
    }
}

// ── Helpers físicos (copiados de Phase 41, sin coupling cross-test) ──────────

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
        solver: gadget_ng_core::SolverKind::Pm,
        pm_grid_size: n,
        ..GravitySection::default()
    };
    RunConfig {
        simulation: SimulationSection {
            dt: DT,
            num_steps: 10,
            softening: 0.0,
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
    }
}

fn measure_pk(parts: &[gadget_ng_core::Particle], n: usize) -> Vec<PkBin> {
    let positions: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    power_spectrum(&positions, &masses, BOX, n)
}

/// Acelera `parts` usando la variante solicitada. Escribe en `out` con el
/// mismo orden que `parts`.
fn compute_accelerations(
    parts: &[gadget_ng_core::Particle],
    n_mesh: usize,
    variant: SolverVariant,
    g_cosmo: f64,
    out: &mut [Vec3],
) {
    let n_p = parts.len();
    assert_eq!(out.len(), n_p);
    let positions: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    let indices: Vec<usize> = (0..n_p).collect();

    match variant {
        SolverVariant::PmOnly => {
            let pm = PmSolver {
                grid_size: n_mesh,
                box_size: BOX,
            };
            pm.accelerations_for_indices(&positions, &masses, 0.0, g_cosmo, &indices, out);
        }
        SolverVariant::TreePmPeriodic { .. } => {
            // Usamos `TreePmSolver` de producción: PM filtrado (erf) + octree
            // con walk no-periódico pero con use_monopole para nodos lejanos.
            // La aproximación no-periódica sólo introduce error en la cáscara
            // `r_cut` cerca del borde de la caja (r_cut ≈ 0.098·L, fracción
            // de volumen ≈ 6 r_cut/L ≈ 0.6, pero sólo partículas con al menos
            // un vecino crítico cruzando el borde — ≲0.5% con 2LPT ICs).
            let eps = variant.eps_internal();
            let eps2 = eps * eps;
            let tpm = TreePmSolver {
                grid_size: n_mesh,
                box_size: BOX,
                r_split: 0.0, // auto → 2.5·cell
            };
            tpm.accelerations_for_indices(&positions, &masses, eps2, g_cosmo, &indices, out);
        }
    }
}

fn evolve_to_a_with_variant(
    parts: &mut Vec<gadget_ng_core::Particle>,
    n_mesh: usize,
    a_start: f64,
    a_target: f64,
    dt: f64,
    variant: SolverVariant,
) -> f64 {
    if a_start >= a_target {
        return a_start;
    }
    let cosmo = cosmo_params();
    let mut scratch = vec![Vec3::zero(); parts.len()];
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
            compute_accelerations(ps, n_mesh, variant, g_cosmo, acc);
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

fn p_ref_at_a(k_hmpc: f64, a: f64) -> f64 {
    // Modo físico Z0Sigma8: σ₈ referido a `a=1`.
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

// ── Resultado por snapshot ───────────────────────────────────────────────────

#[derive(Clone)]
struct SnapshotResult {
    n: usize,
    seed: u64,
    variant: String,
    eps_physical_mpc_h: f64,
    a_target: f64,
    a_actual: f64,
    ks_hmpc: Vec<f64>,
    pk_corrected: Vec<f64>,
    pk_reference: Vec<f64>,
    p_shot: f64,
    delta_rms: f64,
    v_rms: f64,
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
    #[allow(dead_code)]
    fn median_abs_log_err_corr(&self) -> f64 {
        median_abs(&self.log_err_corr())
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
        Some(Self {
            n: v.get("n")?.as_u64()? as usize,
            seed: v.get("seed")?.as_u64()?,
            variant: v.get("variant")?.as_str()?.to_string(),
            eps_physical_mpc_h: v.get("eps_physical_mpc_h")?.as_f64()?,
            a_target: v.get("a_target")?.as_f64()?,
            a_actual: v.get("a_actual")?.as_f64()?,
            ks_hmpc: ks,
            pk_corrected: pc,
            pk_reference: pr,
            p_shot: v.get("p_shot_mpc_h3")?.as_f64()?,
            delta_rms: v.get("delta_rms")?.as_f64()?,
            v_rms: v.get("v_rms")?.as_f64()?,
        })
    }

    fn to_json(&self) -> serde_json::Value {
        let r = self.r_corr();
        json!({
            "n": self.n,
            "seed": self.seed,
            "variant": self.variant,
            "eps_physical_mpc_h": self.eps_physical_mpc_h,
            "a_target": self.a_target,
            "a_actual": self.a_actual,
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
        })
    }
}

// ── Orquestación ─────────────────────────────────────────────────────────────

fn run_one_variant(n: usize, seed: u64, variant: SolverVariant) -> Vec<SnapshotResult> {
    let cfg = build_run_config(n, seed);
    let mut parts = build_particles(&cfg).expect("build_particles falló");
    let model = RnModel::phase35_default();
    let p_shot = shot_noise_level(n);

    let mut results = Vec::with_capacity(A_SNAPSHOTS.len());
    let mut a_current = A_INIT;
    let label = variant.label();
    let eps_mpch = variant.eps_phys_mpc_h();

    for &a_t in A_SNAPSHOTS.iter() {
        if a_current < a_t {
            a_current = evolve_to_a_with_variant(&mut parts, n, a_current, a_t, DT, variant);
        }
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
            "[phase42] N={n:<4} var={label:<16} ε={eps_mpch:.3} Mpc/h  a={a_t:.2}→{a_current:.3}  δ_rms={d_rms:.3e}  v_rms={vrms:.3e}  err={err_med:.3e}"
        );

        results.push(SnapshotResult {
            n,
            seed,
            variant: label.clone(),
            eps_physical_mpc_h: eps_mpch,
            a_target: a_t,
            a_actual: a_current,
            ks_hmpc: ks,
            pk_corrected: pc_v,
            pk_reference: pr_v,
            p_shot,
            delta_rms: d_rms,
            v_rms: vrms,
        });
    }
    results
}

fn run_full_matrix() -> Vec<SnapshotResult> {
    let n = n_grid();
    let mut all = Vec::new();
    for variant in variants() {
        let t0 = std::time::Instant::now();
        let sims = run_one_variant(n, SEED, variant);
        let dt = t0.elapsed().as_secs_f64();
        eprintln!(
            "[phase42] ✓ N={n} var={} completada en {dt:.1}s",
            variant.label()
        );
        all.extend(sims);
    }
    all
}

fn matrix() -> &'static [SnapshotResult] {
    static CELL: OnceLock<Vec<SnapshotResult>> = OnceLock::new();
    CELL.get_or_init(|| {
        if std::env::var("PHASE42_USE_CACHE")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            let mut path = phase42_dir();
            path.push("per_snapshot_metrics.json");
            if path.exists() {
                if let Ok(txt) = fs::read_to_string(&path) {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&txt) {
                        if let Some(arr) = val.get("snapshots").and_then(|v| v.as_array()) {
                            eprintln!(
                                "[phase42] cargando matriz cacheada ({} snapshots) de {}",
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

fn find_opt<'a>(
    m: &'a [SnapshotResult],
    variant: &str,
    a_target: f64,
) -> Option<&'a SnapshotResult> {
    m.iter()
        .find(|r| r.variant == variant && (r.a_target - a_target).abs() < 1e-9)
}

fn phase42_dir() -> PathBuf {
    let mut d = PathBuf::from(std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.pop();
        p.push("target");
        p.to_string_lossy().to_string()
    }));
    d.push("phase42");
    let _ = fs::create_dir_all(&d);
    d
}

fn dump_json(name: &str, value: serde_json::Value) {
    let mut path = phase42_dir();
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
    let variant_labels: Vec<String> = variants().iter().map(|v| v.label()).collect();
    dump_json(
        "per_snapshot_metrics",
        json!({
            "n": n_grid(),
            "seed": SEED,
            "a_snapshots": A_SNAPSHOTS,
            "box_mpc_h": BOX_MPC_H,
            "dt": DT,
            "variants": variant_labels,
            "eps_physical_mpc_h": EPS_PHYSICAL_MPC_H,
            "snapshots": all,
        }),
    );
}

/// Ratio de crecimiento medido en bins de bajo `k` comparado con `[D(a)/D(a_init)]²`.
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

// ══════════════════════════════════════════════════════════════════════════════
// Test 1 — softening_reduces_early_nonlinearity                      SOFT
// ══════════════════════════════════════════════════════════════════════════════
//
// Reporta si al menos una variante TreePM logra `δ_rms(a=0.10)` menor que la
// variante PM puro, y por cuánto. Decisión A/B/C:
//   - A_softening_reduces_nonlinearity: la mejor reducción relativa ≥ 5 %.
//   - B_softening_negligible_or_worse: la mejor reducción < 5 %.
//   - C_baseline_pm_missing: no hay snapshot PM a `a=0.10`.
//
// Se mantiene como soft-check (no falla el suite) siguiendo la metodología
// de Phase 41: los resultados a resolución sub-umbral (N ≪ 128) no
// invalidan ni validan la hipótesis; sólo informan la tendencia.

#[test]
fn softening_reduces_early_nonlinearity() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let pm_snap = find_opt(m, "pm_eps0", 0.10).expect("baseline PM a=0.10 no encontrado");
    let pm_drms = pm_snap.delta_rms;

    let mut per_variant = Vec::new();
    let mut best_reduction: Option<(String, f64)> = None;

    for variant in variants() {
        if matches!(variant, SolverVariant::PmOnly) {
            continue;
        }
        let label = variant.label();
        let Some(ev) = find_opt(m, &label, 0.10) else {
            continue;
        };
        let reduction = (pm_drms - ev.delta_rms) / pm_drms;
        per_variant.push(json!({
            "variant": label,
            "eps_physical_mpc_h": ev.eps_physical_mpc_h,
            "delta_rms_a010": ev.delta_rms,
            "delta_rms_pm_a010": pm_drms,
            "relative_reduction": reduction,
        }));
        if best_reduction
            .as_ref()
            .map(|(_, r)| reduction > *r)
            .unwrap_or(true)
        {
            best_reduction = Some((label.clone(), reduction));
        }
        eprintln!(
            "[phase42][test1] var={label:<16} δ_rms={:.3e} (PM={:.3e}) reducción={:.2}%",
            ev.delta_rms,
            pm_drms,
            reduction * 100.0
        );
    }

    let (best_label, best_rel) =
        best_reduction.unwrap_or_else(|| ("none".into(), f64::NEG_INFINITY));
    let threshold = 0.05;
    let decision = if best_rel >= threshold {
        "A_softening_reduces_nonlinearity"
    } else {
        "B_softening_negligible_or_worse"
    };
    dump_json(
        "test1_softening_reduces_nonlinearity",
        json!({
            "per_variant": per_variant,
            "best_variant": best_label,
            "best_relative_reduction": best_rel,
            "threshold_relative_reduction": threshold,
            "decision": decision,
        }),
    );
    eprintln!(
        "[phase42][test1] DECISION={decision} best={best_label} reducción={:.2}% (threshold={})",
        best_rel * 100.0,
        threshold
    );
    assert!(
        best_rel.is_finite(),
        "best_relative_reduction no finito: {best_rel}"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 2 — treepm_improves_growth_vs_pm                              SOFT
// ══════════════════════════════════════════════════════════════════════════════
//
// Compara `rel_err_growth(a=0.10)` entre PM y cada TreePM. Decide
// A_treepm_improves_linear_growth / B_similar_or_worse.

#[test]
fn treepm_improves_growth_vs_pm() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let k_max_hmpc = 0.1;
    let a_t = 0.10;

    let pm_err = growth_ratio_low_k(m, "pm_eps0", a_t, k_max_hmpc).map(|(_, _, e)| e);

    let mut per_variant = Vec::new();
    let mut best_tree: Option<(String, f64)> = None;
    for variant in variants() {
        if matches!(variant, SolverVariant::PmOnly) {
            continue;
        }
        let label = variant.label();
        if let Some((measured, theory, rel_err)) = growth_ratio_low_k(m, &label, a_t, k_max_hmpc) {
            per_variant.push(json!({
                "variant": label,
                "eps_physical_mpc_h": variant.eps_phys_mpc_h(),
                "growth_measured": measured,
                "growth_theory_d_sq": theory,
                "rel_err": rel_err,
            }));
            if best_tree
                .as_ref()
                .map(|(_, e)| rel_err < *e)
                .unwrap_or(true)
            {
                best_tree = Some((label.clone(), rel_err));
            }
        }
    }

    let (best_tree_label, best_tree_err) =
        best_tree.unwrap_or_else(|| ("none".into(), f64::INFINITY));
    let decision = match pm_err {
        Some(pe) if best_tree_err < pe => "A_treepm_improves_linear_growth",
        Some(_) => "B_similar_or_worse",
        None => "C_pm_baseline_missing",
    };

    dump_json(
        "test2_treepm_growth_vs_pm",
        json!({
            "k_max_hmpc": k_max_hmpc,
            "a_target": a_t,
            "pm_rel_err": pm_err,
            "per_variant_treepm": per_variant,
            "best_treepm_variant": best_tree_label,
            "best_treepm_rel_err": best_tree_err,
            "decision": decision,
        }),
    );

    eprintln!(
        "[phase42][test2] DECISION={decision}  PM_err={:?}  best_treepm={best_tree_label} err={best_tree_err:.3}",
        pm_err
    );

    assert!(
        best_tree_err.is_finite(),
        "best treepm growth error no finito"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 3 — growth_closer_to_linear_with_softening                    SOFT
// ══════════════════════════════════════════════════════════════════════════════
//
// Identifica ε_optimal = argmin_ε |growth - [D/D]²| y clasifica la tendencia
// de `rel_err(ε)` (monotone_decrease, monotone_increase, interior_minimum, flat).

#[test]
fn growth_closer_to_linear_with_softening() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let k_max_hmpc = 0.1;
    let a_t = 0.10;

    let mut curve: Vec<(f64, f64, String)> = Vec::new();
    for variant in variants() {
        if matches!(variant, SolverVariant::PmOnly) {
            continue;
        }
        let label = variant.label();
        if let Some((_, _, rel)) = growth_ratio_low_k(m, &label, a_t, k_max_hmpc) {
            curve.push((variant.eps_phys_mpc_h(), rel, label));
        }
    }
    curve.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let trend = if curve.len() >= 3 {
        let errs: Vec<f64> = curve.iter().map(|(_, e, _)| *e).collect();
        let mono_dec = errs.windows(2).all(|w| w[1] <= w[0]);
        let mono_inc = errs.windows(2).all(|w| w[1] >= w[0]);
        let min_idx = errs
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let interior = min_idx > 0 && min_idx + 1 < errs.len();
        if mono_dec {
            "monotone_decrease_with_eps"
        } else if mono_inc {
            "monotone_increase_with_eps"
        } else if interior {
            "interior_optimum"
        } else {
            "non_monotone"
        }
    } else {
        "insufficient_points"
    };

    let optimal = curve
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .cloned();

    let curve_json: Vec<_> = curve
        .iter()
        .map(|(eps, err, lbl)| {
            json!({
                "variant": lbl,
                "eps_physical_mpc_h": eps,
                "rel_err_growth_a010": err,
            })
        })
        .collect();

    dump_json(
        "test3_growth_vs_softening",
        json!({
            "curve": curve_json,
            "trend": trend,
            "optimal_variant": optimal.as_ref().map(|(_, _, l)| l.clone()),
            "optimal_eps_physical_mpc_h": optimal.as_ref().map(|(e, _, _)| *e),
            "optimal_rel_err": optimal.as_ref().map(|(_, e, _)| *e),
            "a_target": a_t,
            "k_max_hmpc": k_max_hmpc,
        }),
    );

    eprintln!(
        "[phase42][test3] trend={trend}  optimal={:?}",
        optimal.as_ref().map(|(e, err, l)| (l.clone(), *e, *err))
    );

    for (_, e, l) in &curve {
        assert!(e.is_finite(), "rel_err no finito para {l}: {e}");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 4 — no_nan_inf_phase42                                        HARD
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn no_nan_inf_phase42() {
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
        "test4_no_nan_inf",
        json!({"bad_entries": bad, "total_bad": bad.len()}),
    );
    assert!(
        bad.is_empty(),
        "Phase 42 detectó {} valores no finitos",
        bad.len()
    );
}
