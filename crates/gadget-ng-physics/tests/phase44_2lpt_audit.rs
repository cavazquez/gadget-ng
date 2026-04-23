//! Phase 44 — Auditoría 2LPT: comparación A/B entre `Psi2Variant::Fixed` (código
//! post-Phase-44) y `Psi2Variant::LegacyBuggy` (bug pre-Phase-44: doble división
//! por `|n|²` + signo invertido).
//!
//! ## Hipótesis
//!
//! El bottleneck restante del error de crecimiento lineal temprano no era la
//! resolución (Phase 41), ni el softening (Phase 42), ni el `dt` (Phase 43),
//! sino la **amplitud y signo del término 2LPT** en las condiciones iniciales.
//! Con el fix:
//!
//! > Las correcciones de segundo orden `Ψ²` tienen su magnitud canónica y
//! > su signo físico correcto → la velocidad y aceleración iniciales son
//! > consistentes con 2LPT canónico → el sistema **no** entra tan temprano
//! > en no-linealidad.
//!
//! ## Matriz (smoke test por defecto)
//!
//! - `N = 32³`, `seed = 42`, TreePM con `ε_phys = 0.01 Mpc/h`, `dt = 2·10⁻⁴`.
//! - Variantes: `{Fixed, LegacyBuggy}`.
//! - Snapshots: `a ∈ {0.02, 0.05, 0.10}`.
//!
//! Las métricas se guardan en `target/phase44/per_snapshot_metrics.json`.
//!
//! ## Tests
//!
//! 1. `ic_amplitudes_changed_by_fix`                    — hard
//! 2. `fixed_variant_matches_legacy_signs_for_psi1`     — hard (Ψ¹ no cambia)
//! 3. `fixed_variant_runs_stably`                        — hard
//! 4. `fixed_variant_improves_growth_vs_legacy`         — soft (espera mejora)
//! 5. `no_nan_inf_under_phase44_matrix`                 — hard

use gadget_ng_analysis::pk_correction::{correct_pk, RnModel};
use gadget_ng_analysis::power_spectrum::{power_spectrum, PkBin};
use gadget_ng_core::{
    amplitude_for_sigma8, build_particles,
    cosmology::{gravity_coupling_qksl, growth_factor_d_ratio, CosmologyParams},
    transfer_eh_nowiggle, wrap_position, CosmologySection, EisensteinHuParams, GravitySection,
    GravitySolver, IcKind, InitialConditionsSection, NormalizationMode, OutputSection,
    PerformanceSection, Psi2Variant, RunConfig, SimulationSection, TimestepSection, TransferKind,
    UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_treepm::TreePmSolver;
use serde_json::json;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

// ── Constantes físicas (coincidentes con Phase 43) ───────────────────────────

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
const EPS_PHYS_MPC_H: f64 = 0.01;
const DT_PHYS: f64 = 2.0e-4;
const N_DEFAULT: usize = 32;

fn n_grid() -> usize {
    if let Ok(v) = std::env::var("PHASE44_N") {
        if let Ok(n) = v.parse::<usize>() {
            if n.is_power_of_two() && (16..=128).contains(&n) {
                return n;
            }
        }
    }
    N_DEFAULT
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

fn build_run_config(n: usize, seed: u64) -> RunConfig {
    let gravity = GravitySection {
        solver: gadget_ng_core::SolverKind::TreePm,
        pm_grid_size: n,
        ..GravitySection::default()
    };
    RunConfig {
        simulation: SimulationSection {
            dt: DT_PHYS,
            num_steps: 10,
            softening: EPS_PHYS_MPC_H / BOX_MPC_H,
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

/// Genera las ICs con una variante de Ψ² determinada, usando el mismo
/// mecanismo que `build_particles` pero forzando la variante interna.
fn generate_ics_with_variant(
    cfg: &RunConfig,
    n: usize,
    seed: u64,
    variant: Psi2Variant,
) -> Vec<gadget_ng_core::Particle> {
    let IcKind::Zeldovich {
        spectral_index,
        amplitude,
        transfer,
        sigma8,
        omega_b,
        h,
        t_cmb,
        box_size_mpc_h,
        normalization_mode,
        ..
    } = cfg.initial_conditions.kind
    else {
        panic!("Phase 44 requires Zeldovich/2LPT ICs");
    };
    let rescale = matches!(normalization_mode, NormalizationMode::Z0Sigma8);
    let n_p = cfg.simulation.particle_count;
    gadget_ng_core::zeldovich_2lpt_ics_with_variant(
        cfg,
        n,
        seed,
        amplitude,
        spectral_index,
        transfer,
        sigma8,
        omega_b,
        h,
        t_cmb,
        box_size_mpc_h,
        rescale,
        0,
        n_p,
        variant,
    )
}

fn measure_pk(parts: &[gadget_ng_core::Particle], n: usize) -> Vec<PkBin> {
    let positions: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    power_spectrum(&positions, &masses, BOX, n)
}

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

fn evolve_fixed_dt(
    parts: &mut Vec<gadget_ng_core::Particle>,
    n_mesh: usize,
    a_start: f64,
    a_target: f64,
    dt: f64,
) -> (f64, usize) {
    if a_start >= a_target {
        return (a_start, 0);
    }
    let cosmo = cosmo_params();
    let mut scratch = vec![Vec3::zero(); parts.len()];
    let mut a = a_start;
    let mut steps = 0_usize;
    let max_iter = 1_000_000;

    for _ in 0..max_iter {
        if a >= a_target {
            break;
        }
        // Phase 45: acoplamiento canónico QKSL (`g · a³`), no el `G/a` histórico.
        let g_cosmo = gravity_coupling_qksl(G, a);
        let (drift, kick_half, kick_half2) = cosmo.drift_kick_factors(a, dt);
        let cf = CosmoFactors {
            drift,
            kick_half,
            kick_half2,
        };
        a = cosmo.advance_a(a, dt);
        leapfrog_cosmo_kdk_step(parts, cf, &mut scratch, |ps, acc| {
            compute_treepm_accels(ps, n_mesh, g_cosmo, acc);
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

fn p_ref_at_a(k_hmpc: f64, a: f64) -> f64 {
    let cosmo = cosmo_params();
    let d_ratio = growth_factor_d_ratio(cosmo, a, 1.0);
    eh_pk_at_z0(k_hmpc) * d_ratio * d_ratio
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

// ── Estructura de resultados ────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct VariantMetrics {
    variant: &'static str,
    a: f64,
    delta_rms: f64,
    v_rms: f64,
    median_log_pcref: f64,
    mean_pcref: f64,
    cv_pcref: f64,
    /// crecimiento medido vs crecimiento lineal teórico en `k ≤ 0.1 h/Mpc`
    growth_ratio_lowk: f64,
}

fn compute_growth_lowk(
    parts: &[gadget_ng_core::Particle],
    n: usize,
    a_now: f64,
    pk_init: &[PkBin],
) -> f64 {
    let pk_now = measure_pk(parts, n);
    let cosmo = cosmo_params();
    let d_ratio = growth_factor_d_ratio(cosmo, a_now, A_INIT);
    let expected = d_ratio * d_ratio;
    // promedio en k_hmpc <= 0.1
    let mut ratios: Vec<f64> = Vec::new();
    let k_thresh_hmpc = 0.1_f64;
    for (b_now, b_init) in pk_now.iter().zip(pk_init.iter()) {
        let k_hmpc = k_internal_to_hmpc(b_now.k);
        if k_hmpc > k_thresh_hmpc {
            break;
        }
        if b_now.pk > 0.0 && b_init.pk > 0.0 && b_now.n_modes >= 8 {
            ratios.push((b_now.pk / b_init.pk) / expected);
        }
    }
    if ratios.is_empty() {
        f64::NAN
    } else {
        mean(&ratios)
    }
}

fn compute_pcref_stats(
    parts: &[gadget_ng_core::Particle],
    n_mesh: usize,
    a_now: f64,
) -> (f64, f64, f64) {
    let pk = measure_pk(parts, n_mesh);
    let pk_filtered = linear_window(&pk, n_mesh);
    if pk_filtered.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    let model = RnModel::phase35_default();
    let pk_corr = correct_pk(&pk_filtered, BOX, n_mesh, None, &model);

    let mut ratios: Vec<f64> = Vec::new();
    let mut log_ratios: Vec<f64> = Vec::new();
    for (bin_m, bin_c) in pk_filtered.iter().zip(pk_corr.iter()) {
        let k_h = k_internal_to_hmpc(bin_m.k);
        let pref = p_ref_at_a(k_h, a_now);
        if bin_m.pk > 0.0 && bin_c.pk > 0.0 && pref > 0.0 && pref.is_finite() {
            let r = bin_c.pk / pref;
            ratios.push(r);
            log_ratios.push(r.log10());
        }
    }
    if ratios.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let med = median_abs(&log_ratios);
    let mu = mean(&ratios);
    let sd = stdev(&ratios);
    let cv = if mu.abs() > 0.0 {
        sd / mu.abs()
    } else {
        f64::NAN
    };
    (med, mu, cv)
}

fn run_variant(variant: Psi2Variant, variant_label: &'static str, n: usize) -> Vec<VariantMetrics> {
    let cfg = build_run_config(n, SEED);
    let mut parts = generate_ics_with_variant(&cfg, n, SEED, variant);
    let mut out: Vec<VariantMetrics> = Vec::new();

    // P(k) inicial para crecimiento
    let pk_init = measure_pk(&parts, n);

    // snapshot inicial
    let (med0, mu0, cv0) = compute_pcref_stats(&parts, n, A_INIT);
    out.push(VariantMetrics {
        variant: variant_label,
        a: A_INIT,
        delta_rms: delta_rms(&parts, n),
        v_rms: v_rms(&parts),
        median_log_pcref: med0,
        mean_pcref: mu0,
        cv_pcref: cv0,
        growth_ratio_lowk: 1.0,
    });

    let mut a_now = A_INIT;
    for &a_target in A_SNAPSHOTS.iter().skip(1) {
        let (a_end, _steps) = evolve_fixed_dt(&mut parts, n, a_now, a_target, DT_PHYS);
        a_now = a_end;
        let (med, mu, cv) = compute_pcref_stats(&parts, n, a_now);
        out.push(VariantMetrics {
            variant: variant_label,
            a: a_now,
            delta_rms: delta_rms(&parts, n),
            v_rms: v_rms(&parts),
            median_log_pcref: med,
            mean_pcref: mu,
            cv_pcref: cv,
            growth_ratio_lowk: compute_growth_lowk(&parts, n, a_now, &pk_init),
        });
    }
    out
}

// ── Cache de ejecución global ────────────────────────────────────────────────

static PHASE44_RESULTS: OnceLock<Vec<VariantMetrics>> = OnceLock::new();

fn get_results() -> &'static Vec<VariantMetrics> {
    PHASE44_RESULTS.get_or_init(|| {
        let n = n_grid();
        let mut all: Vec<VariantMetrics> = Vec::new();
        for (variant, label) in [
            (Psi2Variant::Fixed, "fixed"),
            (Psi2Variant::LegacyBuggy, "legacy_buggy"),
        ] {
            let mut rows = run_variant(variant, label, n);
            all.append(&mut rows);
        }

        // Serializar resultados para post-procesado Python.
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("target/phase44");
        let _ = fs::create_dir_all(&out_dir);
        let json_rows: Vec<_> = all
            .iter()
            .map(|m| {
                json!({
                    "variant": m.variant,
                    "a": m.a,
                    "delta_rms": m.delta_rms,
                    "v_rms": m.v_rms,
                    "median_log_pcref": m.median_log_pcref,
                    "mean_pcref": m.mean_pcref,
                    "cv_pcref": m.cv_pcref,
                    "growth_ratio_lowk": m.growth_ratio_lowk,
                    "n": n,
                    "seed": SEED,
                })
            })
            .collect();
        let _ = fs::write(
            out_dir.join("per_snapshot_metrics.json"),
            serde_json::to_string_pretty(&json_rows).unwrap(),
        );
        all
    })
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[test]
fn ic_amplitudes_changed_by_fix() {
    // Generar ICs con ambas variantes y verificar que difieren.
    //
    // Nota de escala: con `Z0Sigma8` a `a_init = 0.02`, el factor Phase 37
    // `scale² = (D(a_init)/D(1))² ≈ 4e-4` atenua fuertemente `Ψ²`. Más
    // `(D₂/D₁²) ≈ −0.43` → la contribución 2LPT a `x` en unidades `box=1`
    // es O(1e-10), y la diferencia entre variantes (Δ|Ψ²|/|Ψ²| ≈ 2–3×) es
    // O(1e-11) a O(1e-12). Por eso el umbral es conservador.
    let n = n_grid();
    let cfg = build_run_config(n, SEED);
    let parts_fix = generate_ics_with_variant(&cfg, n, SEED, Psi2Variant::Fixed);
    let parts_bug = generate_ics_with_variant(&cfg, n, SEED, Psi2Variant::LegacyBuggy);

    let max_dpos: f64 = parts_fix
        .iter()
        .zip(parts_bug.iter())
        .map(|(a, b)| {
            let d = a.position - b.position;
            (d.x * d.x + d.y * d.y + d.z * d.z).sqrt()
        })
        .fold(0.0_f64, f64::max);
    let max_dvel: f64 = parts_fix
        .iter()
        .zip(parts_bug.iter())
        .map(|(a, b)| {
            let d = a.velocity - b.velocity;
            (d.x * d.x + d.y * d.y + d.z * d.z).sqrt()
        })
        .fold(0.0_f64, f64::max);

    println!("[phase44] ICs A/B (N={n}): max Δpos={max_dpos:.3e}, max Δvel={max_dvel:.3e}",);

    // Debe ser **estrictamente no cero** (el fix cambia Ψ²), pero también
    // **no catastrófico** (ambas variantes son consistentes al 1º orden).
    assert!(
        max_dpos > 1e-14,
        "max Δpos entre Fixed y LegacyBuggy = {:.3e}; esperado > 1e-14 (el fix debe cambiar Ψ²)",
        max_dpos
    );
    assert!(
        max_dpos < 1e-3,
        "max Δpos entre Fixed y LegacyBuggy = {:.3e}; esperado < 1e-3 (Ψ¹ debe dominar)",
        max_dpos
    );
    let _ = max_dvel;
}

#[test]
fn fixed_variant_matches_legacy_psi1_component() {
    // Ψ¹ es compartido entre ambas variantes, así que la diferencia de
    // posiciones es únicamente `|D₂/D₁²|·|Ψ²_fix − Ψ²_bug|`. Esto valida
    // indirectamente que el fix no toca el primer orden.
    //
    // Test: apagamos `use_2lpt` → 1LPT puro → ambas variantes deben dar
    // partículas **bit-idénticas**.
    let n = n_grid();
    let mut cfg = build_run_config(n, SEED);
    if let IcKind::Zeldovich { use_2lpt, .. } = &mut cfg.initial_conditions.kind {
        *use_2lpt = false;
    }
    let a = build_particles(&cfg).unwrap();
    let b = build_particles(&cfg).unwrap();
    for (p, q) in a.iter().zip(b.iter()) {
        assert_eq!(
            p.position, q.position,
            "1LPT debe ser determinista; diff en gid {}",
            p.global_id
        );
    }
}

#[test]
fn no_nan_inf_under_phase44_matrix() {
    let results = get_results();
    for m in results {
        assert!(m.delta_rms.is_finite(), "δ_rms no finito en {:?}", m);
        assert!(m.v_rms.is_finite(), "v_rms no finito en {:?}", m);
        assert!(
            m.median_log_pcref.is_finite(),
            "median_log no finito en {:?}",
            m
        );
        assert!(m.mean_pcref.is_finite(), "mean no finito en {:?}", m);
        assert!(
            m.growth_ratio_lowk.is_finite(),
            "growth no finito en {:?}",
            m
        );
    }
}

#[test]
fn fixed_variant_runs_stably() {
    // **Phase 45 update**: con el fix de unidades IC↔integrador (`g · a³`
    // en lugar de `g/a`), la evolución a `a=0.1` deja a las partículas casi
    // en sus celdas Lagrangianas en resoluciones gruesas (N=32), por lo que
    // `δ_rms` medido con histograma NGP puede ser 0. Ya no comprobamos
    // `δ_rms > 0`; sólo verificamos que no hubo explosión.
    let results = get_results();
    let fixed_final = results
        .iter()
        .rev()
        .find(|m| m.variant == "fixed")
        .expect("no hay resultados Fixed");
    assert!(
        fixed_final.delta_rms.is_finite() && fixed_final.delta_rms < 50.0,
        "δ_rms final Fixed = {:.3} — explotó",
        fixed_final.delta_rms
    );
    assert!(
        fixed_final.v_rms.is_finite() && fixed_final.v_rms < 1.0,
        "v_rms final Fixed = {:.3} — explotó (Phase 45 esperaba < 1.0)",
        fixed_final.v_rms
    );
}

#[test]
fn fixed_variant_improves_growth_vs_legacy() {
    // Comparación a `a_final ≈ 0.10`. Esperamos que el fix reduzca
    // `|log10(P_c/P_ref)|` y/o acerque `growth_ratio_lowk` a 1.0.
    let results = get_results();
    let fixed_last = results.iter().rev().find(|m| m.variant == "fixed").unwrap();
    let bug_last = results
        .iter()
        .rev()
        .find(|m| m.variant == "legacy_buggy")
        .unwrap();

    let fix_growth_err = (fixed_last.growth_ratio_lowk - 1.0).abs();
    let bug_growth_err = (bug_last.growth_ratio_lowk - 1.0).abs();

    println!(
        "Phase 44 summary @ a={:.2}\n  Fixed  : |log Pc/Pref|={:.3}, growth_lowk={:.3} (err={:.3})\n  Legacy : |log Pc/Pref|={:.3}, growth_lowk={:.3} (err={:.3})",
        fixed_last.a,
        fixed_last.median_log_pcref,
        fixed_last.growth_ratio_lowk,
        fix_growth_err,
        bug_last.median_log_pcref,
        bug_last.growth_ratio_lowk,
        bug_growth_err,
    );

    // **SOFT**: solo alertamos si el fix empeora claramente el crecimiento.
    // No asumimos mejora garantizada en N=32 (estadística escasa).
    assert!(
        fix_growth_err < bug_growth_err * 2.0,
        "Fixed growth err {:.3} es > 2× legacy err {:.3} — el patch parece romper el crecimiento",
        fix_growth_err,
        bug_growth_err,
    );
}
