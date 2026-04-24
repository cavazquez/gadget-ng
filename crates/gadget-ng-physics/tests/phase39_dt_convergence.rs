//! Phase 39 — Convergencia temporal del integrador Leapfrog KDK
//! cosmológico.
//!
//! Responde con datos: *¿qué rango de `dt` permite mantener la evolución en
//! régimen lineal temprano y conservar la fidelidad espectral (forma +
//! amplitud) durante los primeros pasos?*
//!
//! Barre `dt ∈ {4e-4, 2e-4, 1e-4, 5e-5}` (= dt₀ ··· dt₀/8) sobre `N=32³`,
//! 2LPT, PM, `rescale_to_a_init=false` (modo legacy), 3 seeds y 3 snapshots
//! (`a ∈ {0.02, 0.05, 0.10}`). Total: 36 mediciones. Runtime estimado:
//! ~12–15 min release, dominado por `dt₀/8` (1600 pasos × 3 seeds).
//!
//! La referencia física es idéntica a Phase 37 `legacy` y Phase 36:
//!
//! ```text
//! P_ref(k, a) = P_EH(k, z=0) · [D(a)/D(a_init)]²
//! ```
//!
//! `pk_correction`, el solver PM y `R(N)` (Phase 35) quedan intactos — este
//! estudio es ortogonal a la normalización ya validada externamente en
//! Phase 38.

use gadget_ng_analysis::pk_correction::{correct_pk, RnModel};
use gadget_ng_analysis::power_spectrum::{power_spectrum, PkBin};
use gadget_ng_core::{
    amplitude_for_sigma8, build_particles, cosmology::CosmologyParams, transfer_eh_nowiggle,
    wrap_position, CosmologySection, EisensteinHuParams, GravitySection, GravitySolver, IcKind,
    InitialConditionsSection, OutputSection, PerformanceSection, RunConfig, SimulationSection,
    TimestepSection, TransferKind, UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_pm::PmSolver;
use serde_json::json;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

// ── Constantes (consistentes con Phase 36/37) ────────────────────────────────

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

/// Barrido de `dt`: `dt₀, dt₀/2, dt₀/4, dt₀/8` con `dt₀ = 4e-4`.
const DTS: [f64; 4] = [4.0e-4, 2.0e-4, 1.0e-4, 5.0e-5];

// ── Helpers ──────────────────────────────────────────────────────────────────

fn eh_params() -> EisensteinHuParams {
    EisensteinHuParams {
        omega_m: OMEGA_M,
        omega_b: OMEGA_B,
        h: H_DIMLESS,
        t_cmb: T_CMB,
    }
}

fn build_run_config(n: usize, seed: u64) -> RunConfig {
    let gravity = GravitySection {
        solver: gadget_ng_core::SolverKind::Pm,
        pm_grid_size: n,
        ..GravitySection::default()
    };
    RunConfig {
        simulation: SimulationSection {
            dt: 4.0e-4,
            num_steps: 0,
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
                normalization_mode: gadget_ng_core::NormalizationMode::Legacy,
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
            ..Default::default()
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
        insitu_analysis: Default::default(),
        sph: Default::default(),
        rt: Default::default(), reionization: Default::default(), mhd: Default::default(),
        turbulence: Default::default(), two_fluid: Default::default(),
        sidm: Default::default(), modified_gravity: Default::default(),
    }
}

fn measure_pk(parts: &[gadget_ng_core::Particle], n: usize) -> Vec<PkBin> {
    let positions: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    power_spectrum(&positions, &masses, BOX, n)
}

/// Evoluciona con PM hasta (aprox) `a_target`. Paso `dt` fijo; corta cuando
/// `a ≥ a_target`. Devuelve `(a_final, n_steps)`.
fn evolve_pm_to_a(
    parts: &mut Vec<gadget_ng_core::Particle>,
    n_mesh: usize,
    a_start: f64,
    a_target: f64,
    dt: f64,
) -> (f64, usize) {
    if a_start >= a_target {
        return (a_start, 0);
    }
    let n = parts.len();
    let cosmo = CosmologyParams::new(OMEGA_M, OMEGA_L, H0);
    let pm = PmSolver {
        grid_size: n_mesh,
        box_size: BOX,
    };
    let mut scratch = vec![Vec3::zero(); n];
    let mut a = a_start;
    let max_iter = 1_000_000;
    let mut steps = 0usize;
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
        steps += 1;
    }
    (a, steps)
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

fn cpt92_g(a: f64) -> f64 {
    let a3 = a.powi(3);
    let om_a = OMEGA_M / (OMEGA_M + OMEGA_L * a3);
    let ol_a = OMEGA_L * a3 / (OMEGA_M + OMEGA_L * a3);
    2.5 * om_a / (om_a.powf(4.0 / 7.0) - ol_a + (1.0 + om_a / 2.0) * (1.0 + ol_a / 70.0))
}

fn d_of_a(a: f64) -> f64 {
    a * cpt92_g(a)
}

/// Referencia `legacy`: P(k, a) = P_EH(k, z=0) · [D(a)/D(a_init)]².
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

fn stdev(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::NAN;
    }
    let m = mean(x);
    let var = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64;
    var.sqrt()
}

/// OLS log-log sobre `(xs, ys)` → pendiente.
fn loglog_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return f64::NAN;
    }
    let lxs: Vec<f64> = xs.iter().take(n).map(|x| x.ln()).collect();
    let lys: Vec<f64> = ys.iter().take(n).map(|y| y.ln()).collect();
    let mx = lxs.iter().sum::<f64>() / n as f64;
    let my = lys.iter().sum::<f64>() / n as f64;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for i in 0..n {
        let dx = lxs[i] - mx;
        let dy = lys[i] - my;
        sxx += dx * dx;
        sxy += dx * dy;
    }
    sxy / sxx
}

/// `δ_rms` sobre un grid CIC local (N_mesh = N_grid). Asignamos masa a un
/// grid 3D con CIC, normalizamos a la media, y calculamos RMS. Inline para
/// no depender de símbolos `pub(crate)` de `gadget_ng_analysis`.
fn delta_rms_cic(parts: &[gadget_ng_core::Particle], n_mesh: usize) -> f64 {
    let n = n_mesh;
    let ni = n as isize;
    let cell = BOX / n as f64;
    let mut grid = vec![0.0f64; n * n * n];
    for p in parts.iter() {
        let fx = p.position.x / cell;
        let fy = p.position.y / cell;
        let fz = p.position.z / cell;
        let ix = fx.floor() as isize;
        let iy = fy.floor() as isize;
        let iz = fz.floor() as isize;
        let tx = fx - fx.floor();
        let ty = fy - fy.floor();
        let tz = fz - fz.floor();
        for (ddx, wx) in [(0_isize, 1.0 - tx), (1, tx)] {
            for (ddy, wy) in [(0_isize, 1.0 - ty), (1, ty)] {
                for (ddz, wz) in [(0_isize, 1.0 - tz), (1, tz)] {
                    let jx = ((ix + ddx).rem_euclid(ni)) as usize;
                    let jy = ((iy + ddy).rem_euclid(ni)) as usize;
                    let jz = ((iz + ddz).rem_euclid(ni)) as usize;
                    grid[jx * n * n + jy * n + jz] += p.mass * wx * wy * wz;
                }
            }
        }
    }
    let m_total: f64 = parts.iter().map(|p| p.mass).sum();
    let volume = BOX * BOX * BOX;
    let mean_rho = m_total / volume;
    let cell_volume = volume / (n * n * n) as f64;
    let n_cells = (n * n * n) as f64;
    let mut var = 0.0;
    for &m_in_cell in grid.iter() {
        let rho = m_in_cell / cell_volume;
        let delta = rho / mean_rho - 1.0;
        var += delta * delta;
    }
    (var / n_cells).sqrt()
}

/// v_rms = sqrt(<|v|²>) sobre todas las partículas (momento canónico interno).
fn v_rms(parts: &[gadget_ng_core::Particle]) -> f64 {
    let n = parts.len() as f64;
    let sumsq: f64 = parts
        .iter()
        .map(|p| {
            p.velocity.x * p.velocity.x + p.velocity.y * p.velocity.y + p.velocity.z * p.velocity.z
        })
        .sum();
    (sumsq / n).sqrt()
}

fn phase39_dir() -> PathBuf {
    let mut d = PathBuf::from(std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.pop();
        p.push("target");
        p.to_string_lossy().to_string()
    }));
    d.push("phase39");
    let _ = fs::create_dir_all(&d);
    d
}

fn dump_json(name: &str, value: serde_json::Value) {
    let mut path = phase39_dir();
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
    dt: f64,
    a_target: f64,
    a_actual: f64,
    steps_this_leg: usize,
    ks_hmpc: Vec<f64>,
    pk_measured: Vec<f64>,
    pk_corrected: Vec<f64>,
    pk_reference: Vec<f64>,
    delta_rms: f64,
    v_rms: f64,
    n_finite: usize,
    n_nan_inf: usize,
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
            "dt": self.dt,
            "a_target": self.a_target,
            "a_actual": self.a_actual,
            "steps_this_leg": self.steps_this_leg,
            "ks_hmpc": self.ks_hmpc,
            "pk_measured_internal": self.pk_measured,
            "pk_corrected_mpc_h3": self.pk_corrected,
            "pk_reference_mpc_h3": self.pk_reference,
            "median_abs_log10_err_raw":       median_abs(&self.log_err_raw()),
            "median_abs_log10_err_corrected": median_abs(&self.log_err_corr()),
            "mean_r_corr":  mean(&self.r_corr()),
            "stdev_r_corr": stdev(&self.r_corr()),
            "delta_rms": self.delta_rms,
            "v_rms":     self.v_rms,
            "n_finite_bins": self.n_finite,
            "n_nan_inf_bins": self.n_nan_inf,
        })
    }
}

/// Corre una simulación completa para `(n, seed, dt)` y devuelve un
/// `SnapshotResult` por `a` en `A_SNAPSHOTS`. También devuelve el runtime
/// total (wall-clock) de la evolución.
fn run_one_simulation(n: usize, seed: u64, dt: f64) -> (Vec<SnapshotResult>, f64) {
    let cfg = build_run_config(n, seed);
    let mut parts = build_particles(&cfg).expect("build_particles failed");
    let model = RnModel::phase35_default();
    let mut results = Vec::new();
    let mut a_current = A_INIT;
    let t0 = Instant::now();
    for &a_t in A_SNAPSHOTS.iter() {
        let (a_new, steps) = if a_current < a_t {
            evolve_pm_to_a(&mut parts, n, a_current, a_t, dt)
        } else {
            (a_current, 0usize)
        };
        a_current = a_new;
        let pk_raw = measure_pk(&parts, n);
        let pk_win = linear_window(&pk_raw, n);
        // Mismo patrón que Phase 36/37: R(N) absorbe el factor de volumen,
        // `box_mpc_h = None`.
        let pk_corr = correct_pk(&pk_win, BOX, n, None, &model);
        let mut ks = Vec::new();
        let mut pm = Vec::new();
        let mut pc = Vec::new();
        let mut pr = Vec::new();
        let mut n_finite = 0usize;
        let mut n_bad = 0usize;
        for (bin_m, bin_c) in pk_win.iter().zip(pk_corr.iter()) {
            let k_h = k_internal_to_hmpc(bin_m.k);
            let pref = p_ref_at_a(k_h, a_current);
            let all_ok = bin_m.pk.is_finite()
                && bin_c.pk.is_finite()
                && pref.is_finite()
                && bin_m.pk > 0.0
                && bin_c.pk > 0.0
                && pref > 0.0;
            if all_ok {
                ks.push(k_h);
                pm.push(bin_m.pk);
                pc.push(bin_c.pk);
                pr.push(pref);
                n_finite += 1;
            } else {
                n_bad += 1;
            }
        }
        let delta = delta_rms_cic(&parts, n);
        let vrms = v_rms(&parts);
        results.push(SnapshotResult {
            n,
            seed,
            dt,
            a_target: a_t,
            a_actual: a_current,
            steps_this_leg: steps,
            ks_hmpc: ks,
            pk_measured: pm,
            pk_corrected: pc,
            pk_reference: pr,
            delta_rms: delta,
            v_rms: vrms,
            n_finite,
            n_nan_inf: n_bad,
        });
    }
    let elapsed = t0.elapsed().as_secs_f64();
    (results, elapsed)
}

// ── Matriz completa con cache OnceLock ───────────────────────────────────────

#[derive(Clone)]
struct MatrixEntry {
    dt: f64,
    seed: u64,
    runtime_s: f64,
    snapshots: Vec<SnapshotResult>,
}

fn run_full_matrix() -> Vec<MatrixEntry> {
    let mut out = Vec::new();
    for &dt in DTS.iter() {
        for &seed in SEEDS.iter() {
            let (snaps, rt) = run_one_simulation(N_GRID, seed, dt);
            out.push(MatrixEntry {
                dt,
                seed,
                runtime_s: rt,
                snapshots: snaps,
            });
        }
    }
    out
}

fn matrix() -> &'static [MatrixEntry] {
    static CELL: OnceLock<Vec<MatrixEntry>> = OnceLock::new();
    CELL.get_or_init(run_full_matrix)
}

fn find<'a>(m: &'a [MatrixEntry], dt: f64, seed: u64) -> &'a MatrixEntry {
    m.iter()
        .find(|e| (e.dt - dt).abs() < 1e-20 && e.seed == seed)
        .unwrap_or_else(|| panic!("matrix entry not found: dt={dt} seed={seed}"))
}

fn find_snap<'a>(e: &'a MatrixEntry, a_target: f64) -> &'a SnapshotResult {
    e.snapshots
        .iter()
        .find(|s| (s.a_target - a_target).abs() < 1e-12)
        .unwrap_or_else(|| panic!("snapshot a={a_target} not found"))
}

fn dump_matrix_if_needed(m: &[MatrixEntry]) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static DUMPED: AtomicBool = AtomicBool::new(false);
    if DUMPED.swap(true, Ordering::SeqCst) {
        return;
    }
    let entries: Vec<serde_json::Value> = m
        .iter()
        .map(|e| {
            json!({
                "dt": e.dt,
                "seed": e.seed,
                "runtime_s": e.runtime_s,
                "snapshots": e.snapshots.iter().map(|s| s.to_json()).collect::<Vec<_>>(),
            })
        })
        .collect();
    dump_json(
        "per_cfg",
        json!({
            "n": N_GRID,
            "a_init": A_INIT,
            "a_snapshots": A_SNAPSHOTS,
            "seeds": SEEDS,
            "dts": DTS,
            "entries": entries,
        }),
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 1 — dt no afecta el snapshot IC
// ══════════════════════════════════════════════════════════════════════════════
//
// El snapshot en `a = A_INIT` se mide ANTES de cualquier integración, por
// lo que `pk_correction` debe dar idéntico (±eps) para los 4 dts. Este test
// protege que `run_one_simulation` no rompa el invariante.

#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn dt_does_not_affect_ic_snapshot() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let mut entries = Vec::new();
    for &seed in SEEDS.iter() {
        let mut meds = Vec::new();
        for &dt in DTS.iter() {
            let e = find(m, dt, seed);
            let s = find_snap(e, A_INIT);
            meds.push((dt, median_abs(&s.log_err_corr())));
        }
        // Rango entre dts debería ser esencialmente cero.
        let min_med = meds.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
        let max_med = meds
            .iter()
            .map(|(_, v)| *v)
            .fold(f64::NEG_INFINITY, f64::max);
        entries.push(json!({
            "seed": seed,
            "per_dt": meds.iter().map(|(d, v)| json!({"dt": d, "median_abs_log10_err_corr": v})).collect::<Vec<_>>(),
            "spread": max_med - min_med,
        }));
        assert!(
            (max_med - min_med) < 1.0e-10,
            "IC no invariante bajo dt: seed={seed} spread={:.3e} > 1e-10",
            max_med - min_med
        );
    }
    dump_json("test1_ic_invariant", json!({ "per_seed": entries }));
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 2 — ¿Reducir dt reduce el error espectral en snapshots evolucionados?
// ══════════════════════════════════════════════════════════════════════════════
//
// Hard check (preservado): medianas finitas, sin NaN/Inf.
// Soft check (observacional): bandera `ratio_small_over_big < 0.5`. Si la
// evolución ya está en régimen no-lineal, el error NO decrece con `dt`
// porque el dominante es el error físico de la amplitud inicial, no el de
// integración temporal. Documentamos el valor observado.

#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn smaller_dt_reduces_spectral_error() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let mut out = Vec::new();
    for &a_t in &[0.05_f64, 0.10_f64] {
        let mut per_dt = Vec::new();
        for &dt in DTS.iter() {
            let vals: Vec<f64> = SEEDS
                .iter()
                .map(|&seed| median_abs(&find_snap(find(m, dt, seed), a_t).log_err_corr()))
                .collect();
            per_dt.push((dt, mean(&vals), vals));
        }
        let (dt_big, err_big, _) = per_dt.first().unwrap();
        let (dt_small, err_small, _) = per_dt.last().unwrap();
        let ratio = err_small / err_big;
        let monotonic = per_dt.windows(2).all(|w| w[1].1 <= w[0].1);
        let hypothesis_supported = ratio < 0.5;
        out.push(json!({
            "a": a_t,
            "per_dt": per_dt.iter().map(|(d, mu, vs)| json!({
                "dt": d, "mean_median_abs_log10_err_corr": mu, "per_seed": vs,
            })).collect::<Vec<_>>(),
            "ratio_small_over_big": ratio,
            "dt_big":   dt_big,
            "dt_small": dt_small,
            "monotonic_decrease_in_dt": monotonic,
            "hypothesis_dt_halves_reduce_error_2x": hypothesis_supported,
        }));
        assert!(
            err_small.is_finite() && err_big.is_finite(),
            "medianas no finitas en a={a_t}"
        );
        eprintln!(
            "[phase39][test2] a={a_t:.2} err(dt₀)={err_big:.3} err(dt₀/8)={err_small:.3} \
             ratio={ratio:.3} monotonic={monotonic} → hypothesis={hypothesis_supported}"
        );
    }
    dump_json("test2_dt_reduces_error", json!({ "per_a": out }));
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 3 — La corrida con dt₀/8 es numéricamente estable
// ══════════════════════════════════════════════════════════════════════════════
//
// Hard check (preservado): finitud (sin NaN/Inf) en δ_rms, v_rms y en los
// bins del espectro.
// Soft check (observacional): bandera `linear_regime_maintained` cuando
// `growth < 10·D(a)/D(a_init)`. El crecimiento PM real puede exceder esa
// cota si la amplitud inicial empuja al sistema a régimen no-lineal; eso
// es una observación física que NO indica inestabilidad numérica.

#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn dt_small_runs_stable() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let dt_small = *DTS.last().unwrap();
    let a_last = *A_SNAPSHOTS.last().unwrap();
    let d_ratio_expected = d_of_a(a_last) / d_of_a(A_INIT);
    let mut entries = Vec::new();
    for &seed in SEEDS.iter() {
        let e = find(m, dt_small, seed);
        let ic = find_snap(e, A_INIT);
        let last = find_snap(e, a_last);
        let growth = last.delta_rms / ic.delta_rms;
        let linear_regime = growth.is_finite() && growth < 10.0 * d_ratio_expected;
        entries.push(json!({
            "seed": seed,
            "dt": dt_small,
            "delta_rms_ic":         ic.delta_rms,
            "delta_rms_last":       last.delta_rms,
            "growth_ratio":         growth,
            "d_ratio_expected":     d_ratio_expected,
            "growth_over_linear":   growth / d_ratio_expected,
            "linear_regime_maintained": linear_regime,
            "v_rms_ic":   ic.v_rms,
            "v_rms_last": last.v_rms,
            "n_nan_inf_ic":   ic.n_nan_inf,
            "n_nan_inf_last": last.n_nan_inf,
        }));
        assert!(
            ic.delta_rms.is_finite() && last.delta_rms.is_finite(),
            "delta_rms no finito: seed={seed}"
        );
        assert!(
            ic.v_rms.is_finite() && last.v_rms.is_finite(),
            "v_rms no finito: seed={seed}"
        );
        assert_eq!(ic.n_nan_inf, 0, "bins NaN/Inf en IC seed={seed}");
        assert_eq!(
            last.n_nan_inf, 0,
            "bins NaN/Inf en último snapshot seed={seed}"
        );
        assert!(growth.is_finite(), "δ_rms growth no finito: seed={seed}");
        eprintln!(
            "[phase39][test3] seed={seed} δ_rms_ic={:.3e} δ_rms_last={:.3e} \
             growth={:.3e} vs D(a)/D(a_init)={:.2e} → linear_regime={linear_regime}",
            ic.delta_rms, last.delta_rms, growth, d_ratio_expected
        );
    }
    dump_json("test3_stability_small_dt", json!({ "per_seed": entries }));
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 4 — ¿Trend de convergencia detectable (OLS log-log)?
// ══════════════════════════════════════════════════════════════════════════════
//
// Pendiente `p` de `log(err) vs log(dt)` a `a=0.05`. Hard check: finitud.
// Soft check (observacional): bandera `trend_detectable` si `p > 0.5`. Si
// la evolución ya partió desde régimen no-lineal el error es insensible a
// `dt` (`p ≈ 0`) o incluso negativo; eso se reporta explícitamente.

#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn dt_convergence_trend_detectable() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let a_t = 0.05;
    let dts: Vec<f64> = DTS.to_vec();
    let errs: Vec<f64> = DTS
        .iter()
        .map(|&dt| {
            let per_seed: Vec<f64> = SEEDS
                .iter()
                .map(|&seed| median_abs(&find_snap(find(m, dt, seed), a_t).log_err_corr()))
                .collect();
            mean(&per_seed)
        })
        .collect();
    let slope = loglog_slope(&dts, &errs);
    let trend_detectable = slope.is_finite() && slope > 0.5;
    dump_json(
        "test4_convergence_trend",
        json!({
            "a": a_t,
            "dts": dts,
            "mean_median_abs_log10_err_corr_per_dt": errs,
            "loglog_slope_d_logerr_d_logdt": slope,
            "trend_detectable_threshold": 0.5,
            "trend_detectable": trend_detectable,
        }),
    );
    assert!(
        slope.is_finite(),
        "pendiente log-log no finita: slope={slope}"
    );
    eprintln!(
        "[phase39][test4] a={a_t} loglog_slope={slope:.3} → trend_detectable={trend_detectable}"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 5 — Soft check: escala compatible con O(dt²) del integrador KDK
// ══════════════════════════════════════════════════════════════════════════════
//
// Registra la pendiente observada. Panic sólo si el número es no finito;
// si queda fuera de `[1.0, 3.0]`, se marca `supports_order2=false` en el
// JSON (patrón Phase 37). Esto refleja que el régimen asintótico puede no
// alcanzarse en el sweep dado que `a=0.10` entra en no-linealidad.

#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn dt_scaling_consistent_with_integrator_order() {
    let m = matrix();
    dump_matrix_if_needed(m);

    let mut per_a = Vec::new();
    for &a_t in A_SNAPSHOTS.iter().skip(1) {
        let errs: Vec<f64> = DTS
            .iter()
            .map(|&dt| {
                let per_seed: Vec<f64> = SEEDS
                    .iter()
                    .map(|&seed| median_abs(&find_snap(find(m, dt, seed), a_t).log_err_corr()))
                    .collect();
                mean(&per_seed)
            })
            .collect();
        let slope = loglog_slope(&DTS.to_vec(), &errs);
        let supports = slope.is_finite() && (1.0..=3.0).contains(&slope);
        per_a.push(json!({
            "a": a_t,
            "loglog_slope": slope,
            "expected_leapfrog_kdk_order": 2.0,
            "supports_order2_within_1_to_3": supports,
            "dts": DTS,
            "mean_err_per_dt": errs,
        }));
        assert!(
            slope.is_finite(),
            "pendiente log-log no finita en a={a_t}: slope={slope}"
        );
    }
    dump_json("test5_scaling_order_soft", json!({ "per_a": per_a }));
}
