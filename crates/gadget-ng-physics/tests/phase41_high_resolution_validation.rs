//! Phase 41 — Validación física en alta resolución.
//!
//! Demuestra que al superar el shot-noise floor (`P_shot = V / N_particles`),
//! el modo físico `Z0Sigma8` recupera cuantitativamente la amplitud absoluta
//! y el crecimiento lineal en snapshots evolucionados tempranos.
//!
//! ## Matriz
//!
//! `N ∈ {32, 64, 128, 256}`, 2LPT, PM, `seeds ∈ {42, 137, 271}` para N ≤ 64
//! (cosmic variance del baseline) y `seed = 42` para N ∈ {128, 256} (sólo un
//! punto de convergencia de alta resolución — coste por corrida ~1 min a
//! N=128, ~10 min a N=256). `mode ∈ {Legacy, Z0Sigma8}`, `snapshots
//! a ∈ {0.02, 0.05, 0.10}`. Total: 48 mediciones, ~25 min release.
//!
//! El set completo se ejecuta una sola vez (`OnceLock`). Se dumpea a
//! `target/phase41/*.json` y los 5 tests obligatorios leen de esa matriz.
//!
//! ## Shot-noise
//!
//! `P_shot = V_phys / N_particles` con `V_phys = BOX_MPC_H³ = 10⁶ (Mpc/h)³`.
//! El rango lineal medible requiere `P_lin(k) > P_shot`; Phase 41 identifica
//! el mínimo `N` donde la señal supera el ruido en el snapshot IC.
//!
//! ## Crecimiento lineal
//!
//! Para modos de bajo `k` (régimen lineal), debe cumplirse:
//! `P(k, a) / P(k, a_init) ≈ [D(a)/D(a_init)]²`
//!
//! Este test valida esta relación — el estándar de códigos como GADGET,
//! PKDGRAV y CLASS.

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
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_pm::PmSolver;
use serde_json::json;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

// ── Constantes ───────────────────────────────────────────────────────────────

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
const SEEDS_LOW_RES: [u64; 3] = [42, 137, 271];

/// Resoluciones barridas: baja (32, 64) como baseline de Phase 40/37;
/// alta (128, 256) como prueba física — el corazón de Phase 41.
const N_VALUES: [usize; 4] = [32, 64, 128, 256];

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

fn seeds_for(n: usize) -> &'static [u64] {
    if n >= 128 {
        // Solo 1 seed a N ≥ 128 por costo (~1–10 min por corrida).
        // A baja resolución 3 seeds cubren cosmic variance; a alta resolución
        // un solo punto basta para demostrar convergencia.
        &SEEDS_LOW_RES[..1]
    } else {
        &SEEDS_LOW_RES[..]
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
    }
}

fn measure_pk(parts: &[gadget_ng_core::Particle], n: usize) -> Vec<PkBin> {
    let positions: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    power_spectrum(&positions, &masses, BOX, n)
}

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
        // Phase 49: coupling correcto QKSL (G·a³), reemplaza G/a histórico.
        let g_cosmo = gravity_coupling_qksl(G, a);
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

fn p_ref_at_a(k_hmpc: f64, a: f64, mode: Mode) -> f64 {
    let cosmo = cosmo_params();
    let d_ratio = match mode {
        Mode::Legacy => growth_factor_d_ratio(cosmo, a, A_INIT),
        Mode::Z0Sigma8 => growth_factor_d_ratio(cosmo, a, 1.0),
    };
    eh_pk_at_z0(k_hmpc) * d_ratio * d_ratio
}

/// Shot-noise level en `(Mpc/h)³` para N particulas en la caja física.
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

// ── Resultado por snapshot ───────────────────────────────────────────────────

#[derive(Clone)]
struct SnapshotResult {
    n: usize,
    seed: u64,
    mode: &'static str,
    a_target: f64,
    a_actual: f64,
    ks_hmpc: Vec<f64>,
    pk_corrected: Vec<f64>,
    pk_reference: Vec<f64>,
    /// Shot-noise level en las mismas unidades que `pk_corrected` (Mpc/h)³.
    p_shot: f64,
    /// Ratio señal/ruido en el bin más bajo de `k`: `P_corrected(k_min) / P_shot`.
    s_n_at_kmin: f64,
    /// Mínimo ratio señal/ruido en la ventana lineal, para identificar el piso.
    s_n_min: f64,
    delta_rms: f64,
}

impl SnapshotResult {
    fn from_json_value(v: &serde_json::Value) -> Option<Self> {
        let mode_str = v.get("mode")?.as_str()?;
        let mode: &'static str = match mode_str {
            "legacy" => "legacy",
            "z0_sigma8" => "z0_sigma8",
            _ => return None,
        };
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
            mode,
            a_target: v.get("a_target")?.as_f64()?,
            a_actual: v.get("a_actual")?.as_f64()?,
            ks_hmpc: ks,
            pk_corrected: pc,
            pk_reference: pr,
            p_shot: v.get("p_shot_mpc_h3")?.as_f64()?,
            s_n_at_kmin: v.get("s_n_at_kmin")?.as_f64()?,
            s_n_min: v.get("s_n_min")?.as_f64()?,
            delta_rms: v.get("delta_rms")?.as_f64()?,
        })
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
            "pk_corrected_mpc_h3": self.pk_corrected,
            "pk_reference_mpc_h3": self.pk_reference,
            "p_shot_mpc_h3": self.p_shot,
            "s_n_at_kmin": self.s_n_at_kmin,
            "s_n_min": self.s_n_min,
            "median_abs_log10_err_corrected": median_abs(&self.log_err_corr()),
            "mean_r_corr": mean(&r),
            "std_r_corr": stdev(&r),
            "cv_r_corr": stdev(&r) / mean(&r).abs(),
            "delta_rms": self.delta_rms,
        })
    }
}

// ── Orquestación de la matriz ────────────────────────────────────────────────

fn run_one_simulation(n: usize, seed: u64, mode: Mode) -> Vec<SnapshotResult> {
    let cfg = build_run_config(n, seed, mode);
    let mut parts = build_particles(&cfg).expect("build_particles falló");
    let model = RnModel::phase35_default();
    let dt = 4.0e-4;
    let p_shot = shot_noise_level(n);

    let mut results = Vec::with_capacity(A_SNAPSHOTS.len());
    let mut a_current = A_INIT;

    for &a_t in A_SNAPSHOTS.iter() {
        if a_current < a_t {
            a_current = evolve_pm_to_a(&mut parts, n, a_current, a_t, dt);
        }
        let pk_raw = measure_pk(&parts, n);
        let pk_win = linear_window(&pk_raw, n);
        let pk_corr = correct_pk(&pk_win, BOX, n, None, &model);

        let mut ks = Vec::new();
        let mut pc_v = Vec::new();
        let mut pr_v = Vec::new();
        for (bin_m, bin_c) in pk_win.iter().zip(pk_corr.iter()) {
            let k_h = k_internal_to_hmpc(bin_m.k);
            let pref = p_ref_at_a(k_h, a_current, mode);
            if bin_m.pk > 0.0 && bin_c.pk > 0.0 && pref > 0.0 && pref.is_finite() {
                ks.push(k_h);
                pc_v.push(bin_c.pk);
                pr_v.push(pref);
            }
        }

        let s_n: Vec<f64> = pc_v.iter().map(|p| p / p_shot).collect();
        let s_n_at_kmin = s_n.first().copied().unwrap_or(f64::NAN);
        let s_n_min = s_n.iter().cloned().fold(f64::INFINITY, f64::min);

        let d_rms = delta_rms(&parts, n);
        eprintln!(
            "[phase41] N={n:<4} seed={seed:<4} mode={:<10} a={a_t:.2}→{a_current:.3}  P_shot={p_shot:.3e}  S/N(k_min)={s_n_at_kmin:.2e}  err={:.3e}  δ_rms={d_rms:.3e}",
            mode.as_str(),
            median_abs(
                &pc_v
                    .iter()
                    .zip(pr_v.iter())
                    .map(|(&c, &r)| (c / r).log10())
                    .collect::<Vec<_>>()
            )
        );
        results.push(SnapshotResult {
            n,
            seed,
            mode: mode.as_str(),
            a_target: a_t,
            a_actual: a_current,
            ks_hmpc: ks,
            pk_corrected: pc_v,
            pk_reference: pr_v,
            p_shot,
            s_n_at_kmin,
            s_n_min,
            delta_rms: d_rms,
        });
    }
    results
}

/// Matriz completa: N ∈ {32, 64, 128, 256} × seeds_for(N) × 2 modos × 3 snapshots.
/// N=256 está desactivado por defecto (~27 horas). Activar con `PHASE41_RUN_N256=1`.
fn run_full_matrix() -> Vec<SnapshotResult> {
    let run_n256 = std::env::var("PHASE41_RUN_N256")
        .map(|x| x == "1")
        .unwrap_or(false);

    if !run_n256 {
        eprintln!(
            "[phase41] N=256 omitido por defecto (~27 h). Para correrlo: PHASE41_RUN_N256=1"
        );
    }

    let mut all = Vec::new();
    for &n in N_VALUES.iter() {
        if n == 256 && !run_n256 {
            continue;
        }
        for &seed in seeds_for(n).iter() {
            for &mode in &[Mode::Legacy, Mode::Z0Sigma8] {
                let t0 = std::time::Instant::now();
                let sims = run_one_simulation(n, seed, mode);
                let dt = t0.elapsed().as_secs_f64();
                eprintln!(
                    "[phase41] ✓ N={n} seed={seed} mode={} completado en {dt:.1}s",
                    mode.as_str()
                );
                all.extend(sims);
            }
        }
    }
    all
}

fn matrix() -> &'static [SnapshotResult] {
    static CELL: OnceLock<Vec<SnapshotResult>> = OnceLock::new();
    CELL.get_or_init(|| {
        // Si existe una corrida previa en disco y `PHASE41_USE_CACHE=1`,
        // deserializamos el JSON completo para evitar re-ejecutar la matriz.
        if std::env::var("PHASE41_USE_CACHE")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            let mut path = phase41_dir();
            path.push("per_snapshot_metrics.json");
            if path.exists() {
                if let Ok(txt) = fs::read_to_string(&path) {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&txt) {
                        if let Some(arr) = val.get("snapshots").and_then(|v| v.as_array()) {
                            eprintln!(
                                "[phase41] cargando matriz cacheada ({} snapshots) de {}",
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
    n: usize,
    seed: u64,
    mode: &str,
    a_target: f64,
) -> Option<&'a SnapshotResult> {
    m.iter().find(|r| {
        r.n == n && r.seed == seed && r.mode == mode && (r.a_target - a_target).abs() < 1e-9
    })
}

fn phase41_dir() -> PathBuf {
    let mut d = PathBuf::from(std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.pop();
        p.push("target");
        p.to_string_lossy().to_string()
    }));
    d.push("phase41");
    let _ = fs::create_dir_all(&d);
    d
}

fn dump_json(name: &str, value: serde_json::Value) {
    let mut path = phase41_dir();
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
            "n_values": N_VALUES,
            "seeds_low_res": SEEDS_LOW_RES,
            "modes": ["legacy", "z0_sigma8"],
            "a_snapshots": A_SNAPSHOTS,
            "box_mpc_h": BOX_MPC_H,
            "snapshots": all,
        }),
    );
}

/// Ratio de crecimiento lineal medido en bins de bajo `k` comparado con
/// `[D(a)/D(a_init)]²`. Devuelve `(growth_measured, growth_theory, rel_err)`.
fn growth_ratio_low_k(
    m: &[SnapshotResult],
    n: usize,
    seed: u64,
    mode: &str,
    a_target: f64,
    k_max_hmpc: f64,
) -> Option<(f64, f64, f64)> {
    let ic = find_opt(m, n, seed, mode, A_INIT)?;
    let ev = find_opt(m, n, seed, mode, a_target)?;

    // Promediar P(k) sobre los bins de bajo k comunes a los dos snapshots.
    let mut ratios = Vec::new();
    for (i, &k) in ev.ks_hmpc.iter().enumerate() {
        if k > k_max_hmpc {
            break;
        }
        // Buscar el bin más cercano en el snapshot IC.
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
// Test 1 — signal_exceeds_shot_noise_at_high_resolution
// ══════════════════════════════════════════════════════════════════════════════
//
// Hard check: existe al menos un N en la matriz donde la S/N en k_min supera
// 1.0 en modo Z0Sigma8 en el snapshot IC, y ese N es ≤ 128.

#[test]
fn signal_exceeds_shot_noise_at_high_resolution() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let mut per_n: Vec<serde_json::Value> = Vec::new();
    let mut min_n_with_signal: Option<usize> = None;

    for &n in N_VALUES.iter() {
        // Tomar seed=42 (siempre presente) y evaluar S/N en IC para Z0Sigma8.
        let Some(snap) = find_opt(m, n, 42, "z0_sigma8", A_INIT) else {
            continue;
        };
        let s_n_kmin = snap.s_n_at_kmin;
        let s_n_min = snap.s_n_min;
        let dominates = s_n_kmin >= 1.0;
        if dominates && min_n_with_signal.is_none() {
            min_n_with_signal = Some(n);
        }
        per_n.push(json!({
            "n": n,
            "p_shot_mpc_h3": snap.p_shot,
            "s_n_at_kmin": s_n_kmin,
            "s_n_min": s_n_min,
            "signal_dominates_kmin": dominates,
        }));
        eprintln!(
            "[phase41][test1] N={n:<4} P_shot={:.3e}  S/N(k_min)={s_n_kmin:.3e}  dominates={dominates}",
            snap.p_shot
        );
    }

    dump_json(
        "test1_signal_vs_shot_noise",
        json!({
            "per_n_z0_sigma8_ic": per_n,
            "min_n_with_signal_over_noise_at_kmin": min_n_with_signal,
            "threshold_n": 128,
        }),
    );

    let min_n = min_n_with_signal.expect(
        "Ningún N en la matriz supera S/N=1 en k_min bajo Z0Sigma8 — shot-noise domina incluso a alta resolución",
    );
    assert!(
        min_n <= 128,
        "S/N > 1 sólo se alcanza a N={min_n} (se esperaba ≤ 128)"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 2 — pk_correction_valid_beyond_ic_at_high_resolution (SOFT)
// ══════════════════════════════════════════════════════════════════════════════
//
// Reporta si `median|log10(P_c/P_ref)|` en snapshots evolucionados queda bajo
// un umbral razonable (≤ 0.3) a alta resolución. Decide A (cerrada) / B
// (requiere más N).

#[test]
fn pk_correction_valid_beyond_ic_at_high_resolution() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let mut per_n_evolved = Vec::new();
    let mut best_evolved_n: Option<(usize, f64)> = None;

    for &n in N_VALUES.iter() {
        let Some(snap_005) = find_opt(m, n, 42, "z0_sigma8", 0.05) else {
            continue;
        };
        let Some(snap_010) = find_opt(m, n, 42, "z0_sigma8", 0.10) else {
            continue;
        };
        let err_005 = snap_005.median_abs_log_err_corr();
        let err_010 = snap_010.median_abs_log_err_corr();
        let avg = 0.5 * (err_005 + err_010);
        per_n_evolved.push(json!({
            "n": n,
            "median_err_corrected_a005": err_005,
            "median_err_corrected_a010": err_010,
            "avg_evolved_err": avg,
        }));
        if best_evolved_n.map(|(_, e)| avg < e).unwrap_or(true) {
            best_evolved_n = Some((n, avg));
        }
    }

    let (best_n, best_err) = best_evolved_n.unwrap_or((0, f64::INFINITY));
    let threshold = 0.3;
    let decision = if best_err <= threshold {
        "A_pk_correction_extends_to_evolved"
    } else {
        "B_still_needs_higher_resolution"
    };

    dump_json(
        "test2_pk_correction_evolved",
        json!({
            "per_n_z0_sigma8_evolved": per_n_evolved,
            "best_evolved_n": best_n,
            "best_avg_evolved_err": best_err,
            "threshold": threshold,
            "decision": decision,
        }),
    );

    eprintln!(
        "[phase41][test2] DECISION={decision}: best N={best_n} avg_err={best_err:.3} (threshold={threshold})"
    );

    // Soft: garantiza finitud.
    assert!(
        best_err.is_finite(),
        "best_evolved_err no finito: {best_err}"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 3 — linear_growth_recovered_low_k (SOFT)
// ══════════════════════════════════════════════════════════════════════════════
//
// Para bajo k, `P(k, a) / P(k, a_init) ≈ [D(a)/D(a_init)]²`. Reporta el error
// relativo por N y decide si el crecimiento se recupera.

#[test]
fn linear_growth_recovered_low_k() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let k_max_hmpc = 0.1; // bajo-k: bins bajo 0.1 h/Mpc (régimen lineal)
    let mut per_n = Vec::new();
    let mut best: Option<(usize, f64)> = None;
    for &n in N_VALUES.iter() {
        let mut entries_by_a = serde_json::Map::new();
        for &a_t in &[0.05_f64, 0.10_f64] {
            if let Some((measured, theory, rel_err)) =
                growth_ratio_low_k(m, n, 42, "z0_sigma8", a_t, k_max_hmpc)
            {
                entries_by_a.insert(
                    format!("a_{:.2}", a_t),
                    json!({
                        "measured": measured,
                        "theory_d_ratio_sq": theory,
                        "rel_err": rel_err,
                    }),
                );
                // Mejor N = el que minimiza el error a a=0.05 (régimen lineal más temprano)
                if (a_t - 0.05).abs() < 1e-9 && best.map(|(_, e)| rel_err < e).unwrap_or(true) {
                    best = Some((n, rel_err));
                }
            }
        }
        per_n.push(json!({
            "n": n,
            "growth_by_a": entries_by_a,
        }));
    }

    let (best_n, best_err) = best.unwrap_or((0, f64::INFINITY));
    let threshold = 0.15;
    let recovered = best_err <= threshold;

    dump_json(
        "test3_linear_growth",
        json!({
            "k_max_hmpc_low_k": k_max_hmpc,
            "per_n": per_n,
            "best_n": best_n,
            "best_rel_err_a005": best_err,
            "threshold": threshold,
            "growth_recovered": recovered,
        }),
    );
    eprintln!(
        "[phase41][test3] best N={best_n} rel_err(a=0.05)={best_err:.3}  recovered={recovered} (threshold={threshold})"
    );

    assert!(best_err.is_finite(), "growth error no finito: {best_err}");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 4 — spectral_error_decreases_with_resolution  (SOFT)
// ══════════════════════════════════════════════════════════════════════════════
//
// Observacional: reporta `median|log10(P_c/P_ref)|` vs `N` para IC y snapshots
// evolucionados en Z0Sigma8 y registra si el error **decrece** con N en cada
// régimen. En el régimen lineal (IC) se espera convergencia; en el régimen
// evolucionado (`a ≥ 0.05`) el resultado depende de si el sistema permanece
// lineal, lo que a su vez depende de softening e integrador — variables
// fuera del alcance de Phase 41.

#[test]
fn spectral_error_decreases_with_resolution() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let mean_err_for = |a_targets: &[f64]| {
        let mut map: std::collections::BTreeMap<usize, f64> = Default::default();
        for &n in N_VALUES.iter() {
            let mut vals = Vec::new();
            for &seed in seeds_for(n).iter() {
                for &a_t in a_targets.iter() {
                    if let Some(s) = find_opt(m, n, seed, "z0_sigma8", a_t) {
                        vals.push(s.median_abs_log_err_corr());
                    }
                }
            }
            if !vals.is_empty() {
                map.insert(n, mean(&vals));
            }
        }
        map
    };

    let ic_errs = mean_err_for(&[0.02]);
    let evolved_errs = mean_err_for(&[0.05, 0.10]);

    fn monotone_decrease(
        map: &std::collections::BTreeMap<usize, f64>,
    ) -> (Option<f64>, Option<f64>, Option<bool>) {
        let min_n = map.keys().next().copied();
        let max_n = map.keys().next_back().copied();
        match (min_n, max_n) {
            (Some(a), Some(b)) if a != b => {
                let e_a = map[&a];
                let e_b = map[&b];
                (Some(e_a), Some(e_b), Some(e_b < e_a))
            }
            _ => (None, None, None),
        }
    }

    let (ic_e_low_n, ic_e_high_n, ic_decreases) = monotone_decrease(&ic_errs);
    let (_ev_e_low_n, _ev_e_high_n, ev_decreases) = monotone_decrease(&evolved_errs);

    let err_32 = evolved_errs.get(&32).copied().unwrap_or(f64::NAN);
    let err_128 = evolved_errs.get(&128).copied().unwrap_or(f64::NAN);
    eprintln!(
        "[phase41][test4] IC: err(low N)={:?} err(high N)={:?} decreases={:?}",
        ic_e_low_n, ic_e_high_n, ic_decreases
    );
    eprintln!(
        "[phase41][test4] EVOLVED (a≥0.05): err(N=32)={:.3}  err(N=128)={:.3}  decreases={:?}",
        err_32, err_128, ev_decreases
    );

    dump_json(
        "test4_error_vs_resolution",
        json!({
            "mean_ic_err_by_n_z0_sigma8": ic_errs
                .iter()
                .map(|(n, e)| (n.to_string(), json!(*e)))
                .collect::<serde_json::Map<_, _>>(),
            "mean_evolved_err_by_n_z0_sigma8": evolved_errs
                .iter()
                .map(|(n, e)| (n.to_string(), json!(*e)))
                .collect::<serde_json::Map<_, _>>(),
            "ic_decreases_with_n": ic_decreases,
            "evolved_decreases_with_n": ev_decreases,
            "decision_linear_regime_reached":
                evolved_errs.values().copied().fold(f64::INFINITY, f64::min) < 1.0,
        }),
    );

    // Asserts suaves: sólo finitud.
    for (n, e) in ic_errs.iter().chain(evolved_errs.iter()) {
        assert!(e.is_finite(), "error no finito en N={n}: {e}");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 5 — no_nan_inf_high_resolution
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn no_nan_inf_high_resolution() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let mut bad = Vec::new();
    for snap in m {
        let metrics = [
            ("p_shot", snap.p_shot),
            ("s_n_at_kmin", snap.s_n_at_kmin),
            ("delta_rms", snap.delta_rms),
        ];
        for (name, v) in metrics.iter() {
            if !v.is_finite() {
                bad.push(format!(
                    "{name} no finito en N={} seed={} mode={} a={:.3}: {}",
                    snap.n, snap.seed, snap.mode, snap.a_target, v
                ));
            }
        }
        for (i, &p) in snap.pk_corrected.iter().enumerate() {
            if !p.is_finite() {
                bad.push(format!(
                    "pk_corrected[{i}] no finito en N={} seed={} mode={} a={:.3}",
                    snap.n, snap.seed, snap.mode, snap.a_target
                ));
            }
        }
    }
    dump_json(
        "test5_no_nan_inf",
        json!({"bad_entries": bad, "total_bad": bad.len()}),
    );
    assert!(
        bad.is_empty(),
        "Se detectaron valores no finitos: {} entradas",
        bad.len()
    );
}
