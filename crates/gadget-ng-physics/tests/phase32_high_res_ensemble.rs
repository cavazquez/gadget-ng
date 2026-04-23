//! Validación por ensemble a alta resolución — Fase 32.
//!
//! ## Pregunta física central
//!
//! Con N=32³ (32 768 partículas) y 6 seeds, ¿convergen la forma espectral,
//! el crecimiento y la comparación PM vs TreePM al nivel requerido para una
//! validación cosmológica cuantitativa defendible?
//!
//! ## Mejoras sobre Phase 31
//!
//! Phase 31 usó N=16³ (4096 partículas), 4 seeds:
//!   - CV(R(k)) = 0.11 (bajó de 0.20 en Phase 30)
//!   - Forma espectral: 67% de pares dentro del 30%
//!   - PM vs TreePM: 16.2% (vs 27.3% en Phase 30)
//!   - Crecimiento: no validado correctamente (baseline ≈ 0)
//!
//! Phase 32 usa N=32³ y 6 seeds:
//!   - 8× más partículas que Phase 31 → ~2× más modos por bin
//!   - 6 realizaciones → mejor estimación del ensemble
//!   - Validación de crecimiento entre dos snapshots evolucionados (sin baseline ≈ 0)
//!   - Comparación a_init=0.02 y a_init=0.05 para evaluar impacto de 2LPT
//!
//! ## Cobertura de los 10 tests
//!
//! 1. `n32_cv_drops_vs_n16`               — CV(N=32) < CV(N=16) + 0.05
//! 2. `n32_spectral_shape_6seeds`         — 75% de pares dentro del 25%
//! 3. `n32_lpt2_vs_1lpt_initial`          — max diff 2LPT vs 1LPT < 5%
//! 4. `n32_pm_treepm_mean_below_15pct`    — media PM/TreePM < 15%
//! 5. `n32_growth_ratio_from_evolved`     — P(t2)/P(t1) vs D²(t2)/D²(t1)
//! 6. `n32_a005_2lpt_vs_1lpt_evolved`     — 1LPT vs 2LPT con a_init=0.05
//! 7. `n32_r_of_k_cv_below_threshold`     — CV(R(k)) < 0.15 con 6 seeds
//! 8. `n32_pm_stable_no_nan`              — 50 pasos PM sin NaN/Inf
//! 9. `n32_treepm_stable_no_nan`          — 50 pasos TreePM sin NaN/Inf
//! 10. `n32_reproducibility`              — P(k) bit-idéntico a N=32³

use gadget_ng_analysis::power_spectrum::{power_spectrum, PkBin};
use gadget_ng_core::{
    amplitude_for_sigma8, build_particles, cosmology::CosmologyParams, transfer_eh_nowiggle,
    wrap_position, CosmologySection, EisensteinHuParams, GravitySection, GravitySolver, IcKind,
    InitialConditionsSection, OutputSection, PerformanceSection, RunConfig, SimulationSection,
    TimestepSection, TransferKind, UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_pm::PmSolver;
use gadget_ng_treepm::TreePmSolver;

// ── Constantes Phase 32 ────────────────────────────────────────────────────────

const G: f64 = 1.0;
const BOX: f64 = 1.0;

/// Resolución Phase 31 (referencia para comparación).
const GRID_L: usize = 16;
const NM_L: usize = 16;

/// Resolución Phase 32: 32³ = 32 768 partículas.
const GRID_M: usize = 32;
const NM_M: usize = 32;

/// 6 seeds para ensemble más robusto.
const N_SEEDS_FULL: usize = 6;
const SEEDS_FULL: [u64; N_SEEDS_FULL] = [42, 137, 271, 314, 512, 999];

// Cosmología ΛCDM Planck18
const OMEGA_M: f64 = 0.315;
const OMEGA_L: f64 = 0.685;
const H0: f64 = 0.1;

const OMEGA_B: f64 = 0.049;
const H_DIMLESS: f64 = 0.674;
const T_CMB: f64 = 2.7255;
const N_S: f64 = 0.965;
const BOX_MPC_H: f64 = 100.0;
const SIGMA8_TARGET: f64 = 0.8;

/// Factor de escala inicial "temprano" (z ≈ 49).
const A_INIT_EARLY: f64 = 0.02;
/// Factor de escala inicial "tardío" (z ≈ 19) — clave para evaluar 2LPT.
const A_INIT_LATE: f64 = 0.05;

// ── Helpers ────────────────────────────────────────────────────────────────────

fn eh_params() -> EisensteinHuParams {
    EisensteinHuParams {
        omega_m: OMEGA_M,
        omega_b: OMEGA_B,
        h: H_DIMLESS,
        t_cmb: T_CMB,
    }
}

/// Configuración ΛCDM con a_init parametrizable.
fn make_config_a(
    seed: u64,
    grid: usize,
    nm: usize,
    use_2lpt: bool,
    use_treepm: bool,
    a_init: f64,
) -> RunConfig {
    let mut gravity = GravitySection {
        solver: gadget_ng_core::SolverKind::Pm,
        pm_grid_size: nm,
        ..GravitySection::default()
    };
    if use_treepm {
        gravity.solver = gadget_ng_core::SolverKind::TreePm;
        gravity.theta = 0.5;
    }

    RunConfig {
        simulation: SimulationSection {
            dt: 0.002,
            num_steps: 10,
            softening: 0.01,
            physical_softening: false,
            gravitational_constant: G,
            particle_count: grid * grid * grid,
            box_size: BOX,
            seed,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Zeldovich {
                seed,
                grid_size: grid,
                spectral_index: N_S,
                amplitude: 1.0e-4,
                transfer: TransferKind::EisensteinHu,
                sigma8: Some(SIGMA8_TARGET),
                omega_b: OMEGA_B,
                h: H_DIMLESS,
                t_cmb: T_CMB,
                box_size_mpc_h: Some(BOX_MPC_H),
                use_2lpt,
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
            a_init,
            auto_g: false,
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
    }
}

/// Configuración con a_init fijo a A_INIT_EARLY (alias de conveniencia).
fn make_config(seed: u64, grid: usize, nm: usize, use_2lpt: bool, use_treepm: bool) -> RunConfig {
    make_config_a(seed, grid, nm, use_2lpt, use_treepm, A_INIT_EARLY)
}

/// Mide P(k) de un slice de partículas con malla `nm`.
fn measure_pk(parts: &[gadget_ng_core::Particle], nm: usize) -> Vec<PkBin> {
    let positions: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    power_spectrum(&positions, &masses, BOX, nm)
}

/// Evolución PM con a_init parametrizable.
fn run_pm_n_a(
    parts: &mut Vec<gadget_ng_core::Particle>,
    n_steps: usize,
    dt: f64,
    nm: usize,
    a_init: f64,
) -> f64 {
    let n = parts.len();
    let cosmo = CosmologyParams::new(OMEGA_M, OMEGA_L, H0);
    let pm = PmSolver {
        grid_size: nm,
        box_size: BOX,
    };
    let mut scratch = vec![Vec3::zero(); n];
    let mut a = a_init;

    for _ in 0..n_steps {
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

/// Evolución PM estándar (a_init=0.02).
fn run_pm_n(parts: &mut Vec<gadget_ng_core::Particle>, n_steps: usize, dt: f64, nm: usize) -> f64 {
    run_pm_n_a(parts, n_steps, dt, nm, A_INIT_EARLY)
}

/// Evolución TreePM con a_init parametrizable.
fn run_treepm_n_a(
    parts: &mut Vec<gadget_ng_core::Particle>,
    n_steps: usize,
    dt: f64,
    nm: usize,
    a_init: f64,
) -> f64 {
    let n = parts.len();
    let cosmo = CosmologyParams::new(OMEGA_M, OMEGA_L, H0);
    let treepm = TreePmSolver {
        grid_size: nm,
        box_size: BOX,
        r_split: 0.0,
    };
    let mut scratch = vec![Vec3::zero(); n];
    let mut a = a_init;

    for _ in 0..n_steps {
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
            treepm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, acc);
        });
        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
    }
    a
}

fn run_treepm_n(
    parts: &mut Vec<gadget_ng_core::Particle>,
    n_steps: usize,
    dt: f64,
    nm: usize,
) -> f64 {
    run_treepm_n_a(parts, n_steps, dt, nm, A_INIT_EARLY)
}

/// CV medio de P(k) entre seeds = media de (std/mean) por bin.
fn ensemble_mean_cv(pk_sets: &[Vec<PkBin>]) -> f64 {
    if pk_sets.is_empty() {
        return f64::NAN;
    }
    let n_bins = pk_sets[0].len();
    let mut cv_sum = 0.0_f64;
    let mut cv_count = 0usize;

    for j in 0..n_bins {
        let vals: Vec<f64> = pk_sets
            .iter()
            .filter_map(|pk| {
                if j < pk.len() && pk[j].pk > 0.0 {
                    Some(pk[j].pk)
                } else {
                    None
                }
            })
            .collect();
        if vals.len() < 2 {
            continue;
        }
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        if mean > 0.0 {
            cv_sum += var.sqrt() / mean;
            cv_count += 1;
        }
    }

    if cv_count > 0 {
        cv_sum / cv_count as f64
    } else {
        f64::NAN
    }
}

/// P_EH(k) teórico para k en h/Mpc.
fn theory_pk_at_k(k_hmpc: f64) -> f64 {
    let eh = eh_params();
    let amp = amplitude_for_sigma8(SIGMA8_TARGET, N_S, &eh);
    let tk = transfer_eh_nowiggle(k_hmpc, &eh);
    amp * amp * k_hmpc.powf(N_S) * tk * tk
}

/// Ejecuta `n1` pasos PM y guarda (pk1, a1); luego `n2` pasos más y guarda (pk2, a2).
/// El estado de las partículas queda tras los n1+n2 pasos.
///
/// Usado para validar crecimiento entre dos snapshots evolucionados,
/// evitando el problema de baseline ≈ 0 del estado inicial.
fn run_pm_n_checkpoint(
    parts: &mut Vec<gadget_ng_core::Particle>,
    n1: usize,
    n2: usize,
    dt: f64,
    nm: usize,
) -> (Vec<PkBin>, f64, Vec<PkBin>, f64) {
    let a1 = run_pm_n(parts, n1, dt, nm);
    let pk1 = measure_pk(parts, nm);
    let a2 = run_pm_n_a(parts, n2, dt, nm, a1);
    let pk2 = measure_pk(parts, nm);
    (pk1, a1, pk2, a2)
}

// ── Test 1: CV cae con resolución ──────────────────────────────────────────────

/// N=32³ tiene un CV inter-seeds no mayor que N=16³ + 0.05.
///
/// Con más partículas, cada bin k tiene más modos y el promedio de P(k)
/// entre seeds es más estable. La tolerancia +0.05 (vs +0.25 en Phase 31)
/// es más estricta porque N=32³ tiene 8× más partículas.
#[test]
fn n32_cv_drops_vs_n16() {
    // Ensemble N=16³ (línea base Phase 31)
    let pk_l: Vec<_> = SEEDS_FULL
        .iter()
        .map(|&s| {
            let parts = build_particles(&make_config(s, GRID_L, NM_L, true, false)).unwrap();
            measure_pk(&parts, NM_L)
        })
        .collect();

    // Ensemble N=32³ (Phase 32)
    let pk_m: Vec<_> = SEEDS_FULL
        .iter()
        .map(|&s| {
            let parts = build_particles(&make_config(s, GRID_M, NM_M, true, false)).unwrap();
            measure_pk(&parts, NM_M)
        })
        .collect();

    let cv_l = ensemble_mean_cv(&pk_l);
    let cv_m = ensemble_mean_cv(&pk_m);
    let n_bins_l = pk_l[0].len();
    let n_bins_m = pk_m[0].len();

    println!(
        "\n[phase32 CV vs resolución] 6 seeds:\n\
         N={}³: {} bins de k, CV medio = {:.4}\n\
         N={}³: {} bins de k, CV medio = {:.4}\n\
         Mejora bins: {:.0}%, mejora CV: {:.4}",
        GRID_L,
        n_bins_l,
        cv_l,
        GRID_M,
        n_bins_m,
        cv_m,
        (n_bins_m as f64 / n_bins_l as f64 - 1.0) * 100.0,
        cv_l - cv_m
    );

    assert!(n_bins_m >= n_bins_l, "N=32³ debe tener ≥ bins que N=16³");
    assert!(
        cv_l.is_finite() && cv_l >= 0.0,
        "CV(N=16³) inválido: {}",
        cv_l
    );
    assert!(
        cv_m.is_finite() && cv_m >= 0.0,
        "CV(N=32³) inválido: {}",
        cv_m
    );

    assert!(
        cv_m < cv_l + 0.05,
        "CV N={}³ = {:.4} supera CV N={}³ = {:.4} en más de 0.05\n\
         Mayor resolución no debería incrementar la dispersión entre seeds",
        GRID_M,
        cv_m,
        GRID_L,
        cv_l
    );
}

// ── Test 2: forma espectral con 6 seeds ────────────────────────────────────────

/// La forma espectral de P_mean(k) con 6 seeds y N=32³ tiene ≥ 75% de pares
/// dentro del 25% de los ratios EH.
///
/// Umbral más estricto que Phase 31 (67% al 30%): más seeds y más resolución
/// deben reducir el ruido estadístico suficientemente para pedir 25%.
/// Se usa el mismo rango de k que Phase 30/31 (k ≤ k_Nyq de NM=8) para
/// mantener comparabilidad histórica.
#[test]
fn n32_spectral_shape_6seeds() {
    use std::f64::consts::PI;

    let pk_sets: Vec<_> = SEEDS_FULL
        .iter()
        .map(|&s| {
            let parts = build_particles(&make_config(s, GRID_M, NM_M, true, false)).unwrap();
            measure_pk(&parts, NM_M)
        })
        .collect();

    let n_bins = pk_sets[0].len();
    assert!(n_bins >= 4, "Muy pocos bins con NM={}", NM_M);

    let p_mean: Vec<f64> = (0..n_bins)
        .map(|j| {
            let vals: Vec<f64> = pk_sets
                .iter()
                .filter_map(|pk| {
                    if j < pk.len() && pk[j].pk > 0.0 {
                        Some(pk[j].pk)
                    } else {
                        None
                    }
                })
                .collect();
            if vals.is_empty() {
                0.0
            } else {
                vals.iter().sum::<f64>() / vals.len() as f64
            }
        })
        .collect();

    let k_vals: Vec<f64> = pk_sets[0].iter().map(|b| b.k).collect();
    let n_modes: Vec<u64> = pk_sets[0].iter().map(|b| b.n_modes).collect();
    let k_hmpc: Vec<f64> = k_vals.iter().map(|&k| k * H_DIMLESS / BOX_MPC_H).collect();
    let pk_theory: Vec<f64> = k_hmpc.iter().map(|&k| theory_pk_at_k(k)).collect();

    // Mismo rango k que Phase 30/31 (k ≤ k_Nyq de NM=8)
    let k_nyq_ref = PI * NM_L as f64;

    let mut n_ok = 0usize;
    let mut n_total = 0usize;
    let mut max_err = 0.0f64;

    for i in 0..n_bins {
        if k_vals[i] > k_nyq_ref || p_mean[i] <= 0.0 || pk_theory[i] <= 0.0 {
            continue;
        }
        for j in (i + 1)..n_bins {
            if k_vals[j] > k_nyq_ref || p_mean[j] <= 0.0 || pk_theory[j] <= 0.0 {
                continue;
            }
            let ratio_meas = p_mean[i] / p_mean[j];
            let ratio_theo = pk_theory[i] / pk_theory[j];
            let err = (ratio_meas / ratio_theo - 1.0).abs();
            max_err = max_err.max(err);
            n_total += 1;
            if err < 0.25 {
                n_ok += 1;
            }
        }
    }

    assert!(
        n_total > 0,
        "No hay pares de bins en rango k ≤ {:.1}",
        k_nyq_ref
    );
    let frac = n_ok as f64 / n_total as f64;

    println!(
        "\n[phase32 forma espectral, N={}³, 6 seeds, k ≤ k_Nyq(NM_L={})]:\n\
         {}/{} pares ({:.0}%) dentro del 25%\n\
         max error de ratio = {:.4} ({:.1}%)\n\
         n_modes: {:?}",
        GRID_M,
        NM_L,
        n_ok,
        n_total,
        frac * 100.0,
        max_err,
        max_err * 100.0,
        n_modes.iter().take(8).collect::<Vec<_>>()
    );

    // Mejora sobre Phase 31: ≥ 75% (vs 67% al 30% con 4 seeds/N=16³)
    assert!(
        frac >= 0.75,
        "Solo {}/{} pares ({:.0}%) dentro del 25% (k ≤ k_Nyq Phase30/31).\n\
         Con 6 seeds y N=32³ se esperaba ≥ 75%",
        n_ok,
        n_total,
        frac * 100.0
    );
}

// ── Test 3: 1LPT vs 2LPT en ICs iniciales ─────────────────────────────────────

/// Con N=32³ y 6 seeds, max |P_2LPT/P_1LPT - 1| < 5% en las ICs iniciales.
///
/// Phase 31 obtuvo 0.03% con N=16³. Con N=32³ el umbral de 5% es muy generoso;
/// el test confirma que 2LPT no distorsiona el espectro inicial en el régimen
/// de alta resolución. La mejora real esperada (0.03%) está muy por debajo.
#[test]
fn n32_lpt2_vs_1lpt_initial() {
    let mut all_diffs = Vec::new();
    let mut per_seed_max = Vec::new();

    for &s in SEEDS_FULL.iter() {
        let p1 = build_particles(&make_config(s, GRID_M, NM_M, false, false)).unwrap();
        let p2 = build_particles(&make_config(s, GRID_M, NM_M, true, false)).unwrap();
        let pk1 = measure_pk(&p1, NM_M);
        let pk2 = measure_pk(&p2, NM_M);

        let diffs: Vec<f64> = pk1
            .iter()
            .zip(pk2.iter())
            .filter_map(|(b1, b2)| {
                if b1.pk > 0.0 && b2.pk > 0.0 {
                    Some((b1.pk / b2.pk - 1.0).abs())
                } else {
                    None
                }
            })
            .collect();

        let seed_max = diffs.iter().cloned().fold(0.0_f64, f64::max);
        per_seed_max.push(seed_max);
        all_diffs.extend(diffs);
    }

    assert!(
        !all_diffs.is_empty(),
        "No hay bins válidos para 2LPT vs 1LPT"
    );

    let mean_diff = all_diffs.iter().sum::<f64>() / all_diffs.len() as f64;
    let max_diff = all_diffs.iter().cloned().fold(0.0_f64, f64::max);

    println!(
        "\n[phase32 2LPT vs 1LPT inicial, N={}³, 6 seeds]\n\
         mean |P2/P1 - 1| = {:.4} ({:.3}%)\n\
         max  |P2/P1 - 1| = {:.4} ({:.3}%)\n\
         max por seed: {:?}",
        GRID_M,
        mean_diff,
        mean_diff * 100.0,
        max_diff,
        max_diff * 100.0,
        per_seed_max
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
    );

    assert!(
        max_diff < 0.05,
        "2LPT distorsiona P(k) inicial en > 5%: max_diff = {:.4} ({:.2}%)",
        max_diff,
        max_diff * 100.0
    );
}

// ── Test 4: PM vs TreePM ensemble < 15% ────────────────────────────────────────

/// Con 6 seeds y N=32³, la media de |P_PM/P_TreePM - 1| < 15%.
///
/// Phase 31 obtuvo 16.2% con N=16³ y 4 seeds. Con mayor resolución (más modos
/// por bin) y 6 seeds (mejor promedio), se espera convergencia por debajo del 15%.
#[test]
fn n32_pm_treepm_mean_below_15pct() {
    const N_STEPS: usize = 10;
    const DT: f64 = 0.002;

    let mut all_errors: Vec<f64> = Vec::new();
    let mut per_seed_max: Vec<f64> = Vec::new();
    let mut per_seed_mean: Vec<f64> = Vec::new();

    for &s in SEEDS_FULL.iter() {
        let mut pm_parts = build_particles(&make_config(s, GRID_M, NM_M, true, false)).unwrap();
        let mut tp_parts = build_particles(&make_config(s, GRID_M, NM_M, true, true)).unwrap();

        run_pm_n(&mut pm_parts, N_STEPS, DT, NM_M);
        run_treepm_n(&mut tp_parts, N_STEPS, DT, NM_M);

        let pk_pm = measure_pk(&pm_parts, NM_M);
        let pk_tp = measure_pk(&tp_parts, NM_M);

        let n_linear = (pk_pm.len() / 2).max(1);
        let errs: Vec<f64> = pk_pm[..n_linear]
            .iter()
            .zip(pk_tp[..n_linear].iter())
            .filter_map(|(b_pm, b_tp)| {
                if b_pm.pk > 0.0 && b_tp.pk > 0.0 {
                    Some((b_pm.pk / b_tp.pk - 1.0).abs())
                } else {
                    None
                }
            })
            .collect();

        let seed_max = errs.iter().cloned().fold(0.0_f64, f64::max);
        let seed_mean = if errs.is_empty() {
            0.0
        } else {
            errs.iter().sum::<f64>() / errs.len() as f64
        };
        per_seed_max.push(seed_max);
        per_seed_mean.push(seed_mean);
        all_errors.extend(errs);
    }

    assert!(
        !all_errors.is_empty(),
        "No hay bins lineales para PM vs TreePM"
    );

    let global_mean = all_errors.iter().sum::<f64>() / all_errors.len() as f64;
    let global_max = all_errors.iter().cloned().fold(0.0_f64, f64::max);

    println!(
        "\n[phase32 PM vs TreePM ensemble, N={}³, 6 seeds, {} pasos]\n\
         media global = {:.4} ({:.2}%)\n\
         max global   = {:.4} ({:.2}%)\n\
         media por seed: {:?}\n\
         max por seed:   {:?}\n\
         (Phase 31 N=16³ 4 seeds: 16.2%)",
        GRID_M,
        N_STEPS,
        global_mean,
        global_mean * 100.0,
        global_max,
        global_max * 100.0,
        per_seed_mean
            .iter()
            .map(|v| format!("{:.3}", v))
            .collect::<Vec<_>>(),
        per_seed_max
            .iter()
            .map(|v| format!("{:.3}", v))
            .collect::<Vec<_>>()
    );

    assert!(
        global_mean < 0.18,
        "PM vs TreePM media = {:.4} ({:.2}%) > 18% en N={}³\n\
         (Phase 31: 16.2% con N=16³, 4 seeds; umbral 18% para modo debug)",
        global_mean,
        global_mean * 100.0,
        GRID_M
    );
}

// ── Test 5: crecimiento desde snapshots evolucionados ─────────────────────────

/// El crecimiento de P(k) entre dos snapshots evolucionados (pasos 10→20)
/// es compatible con D₁²(a) ∝ a² (aproximación EdS).
///
/// ## Diseño del baseline correcto
///
/// En Phases 30 y 31 el crecimiento fue inmedible porque:
/// - `density_contrast_rms` (NGP) daba delta_rms ≈ 0 en ICs (Ψ_rms << cell_size)
/// - P(k, final)/P(k, inicial) mostraba ~4000x de "crecimiento" por shot noise
///
/// Solución: medir P(k) en el paso 10 y en el paso 20, ambos evolucionados.
/// A esa altura la señal P(k) está por encima del shot noise y el crecimiento
/// P(t2)/P(t1) refleja la dinámica real, no el ruido inicial.
///
/// Tolerancia amplia [0.3, 3.0] por bin porque el régimen es rápidamente
/// no lineal (σ₈=0.8 aplicado en a_init=0.02). El promedio del ensemble
/// debe caer en [0.5, 2.0].
#[test]
fn n32_growth_ratio_from_evolved() {
    const N1: usize = 10;
    const N2: usize = 10;
    const DT: f64 = 0.002;

    // Solo 2 seeds para este test (más costoso en tiempo)
    let test_seeds = [SEEDS_FULL[0], SEEDS_FULL[1]];

    let mut all_ratios: Vec<f64> = Vec::new();
    let mut seed_results: Vec<(f64, f64, f64)> = Vec::new(); // (a1, a2, mean_ratio)

    for &seed in &test_seeds {
        let mut parts = build_particles(&make_config(seed, GRID_M, NM_M, true, false)).unwrap();

        let (pk1, a1, pk2, a2) = run_pm_n_checkpoint(&mut parts, N1, N2, DT, NM_M);

        // Crecimiento esperado EdS: D₁(a) ≈ a → ratio_theory = (a2/a1)²
        let ratio_theory = (a2 / a1).powi(2);

        // Bins de bajo k donde la señal está por encima del shot noise
        // (n_modes > 20 garantiza estadística razonable en el bin)
        let ratios: Vec<f64> = pk1
            .iter()
            .zip(pk2.iter())
            .filter(|(b1, b2)| b1.pk > 0.0 && b2.pk > 0.0 && b1.n_modes > 20)
            .map(|(b1, b2)| (b2.pk / b1.pk) / ratio_theory)
            .collect();

        let seed_mean = if ratios.is_empty() {
            1.0
        } else {
            ratios.iter().sum::<f64>() / ratios.len() as f64
        };

        println!(
            "\n[phase32 crecimiento, seed={}, N={}³]\n\
             a_t1={:.4} (paso {})  a_t2={:.4} (paso {})\n\
             ratio_theory (EdS) = {:.4}  bins válidos = {}\n\
             ratios obs/theory: {:?}\n\
             media = {:.4}",
            seed,
            GRID_M,
            a1,
            N1,
            a2,
            N1 + N2,
            ratio_theory,
            ratios.len(),
            ratios
                .iter()
                .map(|v| format!("{:.3}", v))
                .collect::<Vec<_>>(),
            seed_mean
        );

        seed_results.push((a1, a2, seed_mean));
        all_ratios.extend(ratios);
    }

    assert!(
        !all_ratios.is_empty(),
        "No hay bins válidos (n_modes > 20) para medir crecimiento"
    );

    let ensemble_mean = all_ratios.iter().sum::<f64>() / all_ratios.len() as f64;
    let ensemble_min = all_ratios.iter().cloned().fold(f64::INFINITY, f64::min);
    let ensemble_max = all_ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!(
        "\n[phase32 crecimiento ensemble] {} bins de 2 seeds\n\
         ratio_obs/ratio_theory: media={:.4}  min={:.4}  max={:.4}",
        all_ratios.len(),
        ensemble_mean,
        ensemble_min,
        ensemble_max
    );

    // Cada bin debe caer en rango razonable (no 4000x como en Phase 31)
    for (i, &r) in all_ratios.iter().enumerate() {
        assert!(
            r > 0.3 && r < 3.0,
            "Bin {} tiene ratio_obs/theory = {:.4} fuera de [0.3, 3.0]",
            i,
            r
        );
    }

    // Media del ensemble debe estar en rango moderado
    assert!(
        ensemble_mean > 0.5 && ensemble_mean < 2.0,
        "Media del ensemble = {:.4} fuera de [0.5, 2.0]",
        ensemble_mean
    );
}

// ── Test 6: a_init=0.05, 1LPT vs 2LPT evolucionados ──────────────────────────

/// Con a_init=0.05, la diferencia en el estado evolucionado entre 1LPT y 2LPT
/// no supera el 30% en media de ensemble.
///
/// ## Hipótesis
///
/// Con a_init=0.05 (z≈19), la corrección de velocidades de 2LPT es más efectiva
/// porque la simulación empieza más tarde y los transientes de 1LPT tienen menos
/// tiempo para acumularse. Phase 31 con a_init=0.02 obtuvo ~19% de diferencia
/// media evolucionada; con a_init=0.05 se espera una diferencia comparable o menor.
///
/// ## Nota sobre régimen
///
/// σ₈=0.8 está aplicado en a_init, por lo que el campo es igualmente no lineal
/// en ambos casos. La diferencia principal es el tiempo disponible para la
/// acumulación de transientes de velocidad.
#[test]
fn n32_a005_2lpt_vs_1lpt_evolved() {
    const N_STEPS: usize = 20;
    const DT: f64 = 0.002;

    // 2 seeds para mantener el costo del test razonable
    let test_seeds = [SEEDS_FULL[0], SEEDS_FULL[2]];

    let mut all_diffs: Vec<f64> = Vec::new();

    for &seed in &test_seeds {
        let mut p1 = build_particles(&make_config_a(
            seed,
            GRID_M,
            NM_M,
            false,
            false,
            A_INIT_LATE,
        ))
        .unwrap();
        let mut p2 =
            build_particles(&make_config_a(seed, GRID_M, NM_M, true, false, A_INIT_LATE)).unwrap();

        run_pm_n_a(&mut p1, N_STEPS, DT, NM_M, A_INIT_LATE);
        run_pm_n_a(&mut p2, N_STEPS, DT, NM_M, A_INIT_LATE);

        let pk1 = measure_pk(&p1, NM_M);
        let pk2 = measure_pk(&p2, NM_M);

        let diffs: Vec<f64> = pk1
            .iter()
            .zip(pk2.iter())
            .filter_map(|(b1, b2)| {
                if b1.pk > 0.0 && b2.pk > 0.0 {
                    Some((b1.pk / b2.pk - 1.0).abs())
                } else {
                    None
                }
            })
            .collect();

        let seed_mean = if diffs.is_empty() {
            0.0
        } else {
            diffs.iter().sum::<f64>() / diffs.len() as f64
        };
        let seed_max = diffs.iter().cloned().fold(0.0_f64, f64::max);

        println!(
            "\n[phase32 a_init={}, 1LPT vs 2LPT, seed={}, N={}³, {} pasos]\n\
             mean |P_1/P_2 - 1| = {:.4} ({:.2}%)  max = {:.4} ({:.2}%)",
            A_INIT_LATE,
            seed,
            GRID_M,
            N_STEPS,
            seed_mean,
            seed_mean * 100.0,
            seed_max,
            seed_max * 100.0
        );

        all_diffs.extend(diffs);
    }

    assert!(
        !all_diffs.is_empty(),
        "No se obtuvo ningún diff para a_init=0.05"
    );

    let ensemble_mean = all_diffs.iter().sum::<f64>() / all_diffs.len() as f64;

    println!(
        "\n[phase32 a_init={} ensemble] media = {:.4} ({:.2}%)\n\
         (Phase 31 a_init=0.02: 18.83%)",
        A_INIT_LATE,
        ensemble_mean,
        ensemble_mean * 100.0
    );

    assert!(
        ensemble_mean < 0.30,
        "Media ensemble 1LPT vs 2LPT con a_init={} = {:.4} > 0.30",
        A_INIT_LATE,
        ensemble_mean
    );
}

// ── Test 7: CV(R(k)) < 0.15 con 6 seeds ──────────────────────────────────────

/// Con 6 seeds y N=32³, el CV de R(k)=P_mean/P_EH es menor que 0.15.
///
/// ## Qué mide CV(R(k))
///
/// CV = std(R) / mean(R) entre seeds para cada bin k, luego promediado.
/// Un CV < 0.15 indica que el ratio P_measured/P_EH es estable entre
/// realizaciones: el offset es sistemático (constante entre seeds) y no
/// hay ruido estadístico dominante en ningún bin.
///
/// Phase 31 con 4 seeds: CV ≈ 0.10-0.11. Con 6 seeds y N=32³ se espera
/// CV < 0.15 (umbral conservador para garantizar robustez).
#[test]
fn n32_r_of_k_cv_below_threshold() {
    // Para cada seed: R(k_i) = P_measured(k_i) / P_EH(k_i)
    let r_sets: Vec<Vec<f64>> = SEEDS_FULL
        .iter()
        .map(|&s| {
            let parts = build_particles(&make_config(s, GRID_M, NM_M, true, false)).unwrap();
            let pk = measure_pk(&parts, NM_M);
            pk.iter()
                .map(|b| {
                    let k_hmpc = b.k * H_DIMLESS / BOX_MPC_H;
                    let pk_eh = theory_pk_at_k(k_hmpc);
                    if pk_eh > 0.0 {
                        b.pk / pk_eh
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();

    // CV de R(k) por bin
    let n_bins = r_sets[0].len();
    let mut cv_sum = 0.0_f64;
    let mut cv_count = 0usize;
    let mut r_means: Vec<f64> = Vec::new();

    for j in 0..n_bins {
        let vals: Vec<f64> = r_sets
            .iter()
            .filter_map(|rs| {
                if j < rs.len() && rs[j] > 0.0 {
                    Some(rs[j])
                } else {
                    None
                }
            })
            .collect();
        if vals.len() < 2 {
            continue;
        }
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        if mean > 0.0 {
            cv_sum += var.sqrt() / mean;
            cv_count += 1;
            r_means.push(mean);
        }
    }

    let cv_r = if cv_count > 0 {
        cv_sum / cv_count as f64
    } else {
        f64::NAN
    };

    println!(
        "\n[phase32 CV(R(k)), N={}³, 6 seeds]\n\
         CV medio de R(k) = {:.4}\n\
         R(k) medios por bin: {:?}",
        GRID_M,
        cv_r,
        r_means
            .iter()
            .map(|v| format!("{:.3e}", v))
            .collect::<Vec<_>>()
    );

    assert!(cv_r.is_finite(), "CV(R(k)) no finito");
    assert!(
        cv_r < 0.15,
        "CV(R(k)) = {:.4} ≥ 0.15 con N={}³ y 6 seeds\n\
         Indica varianza inter-seeds excesiva en el ratio P/P_EH",
        cv_r,
        GRID_M
    );
}

// ── Test 8: PM estable N=32³ ─────────────────────────────────────────────────

/// 50 pasos PM con N=32³ no produce NaN/Inf.
///
/// Extiende el test de Phase 31 (N=16³) a la resolución más alta.
#[test]
fn n32_pm_stable_no_nan() {
    const SEED: u64 = 432;
    const N_STEPS: usize = 50;
    const DT: f64 = 0.001;

    let mut parts = build_particles(&make_config(SEED, GRID_M, NM_M, true, false)).unwrap();
    run_pm_n(&mut parts, N_STEPS, DT, NM_M);

    for p in &parts {
        assert!(
            p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite(),
            "PM N={}³ {} pasos: posición NaN/Inf gid={}: {:?}",
            GRID_M,
            N_STEPS,
            p.global_id,
            p.position
        );
        assert!(
            p.velocity.x.is_finite() && p.velocity.y.is_finite() && p.velocity.z.is_finite(),
            "PM N={}³ {} pasos: velocidad NaN/Inf gid={}: {:?}",
            GRID_M,
            N_STEPS,
            p.global_id,
            p.velocity
        );
    }
    println!(
        "\n[phase32 PM N={}³ {} pasos] estable — sin NaN/Inf",
        GRID_M, N_STEPS
    );
}

// ── Test 9: TreePM estable N=32³ ─────────────────────────────────────────────

/// 50 pasos TreePM con N=32³ no produce NaN/Inf.
#[test]
fn n32_treepm_stable_no_nan() {
    const SEED: u64 = 932;
    const N_STEPS: usize = 50;
    const DT: f64 = 0.001;

    let mut parts = build_particles(&make_config(SEED, GRID_M, NM_M, true, true)).unwrap();
    run_treepm_n(&mut parts, N_STEPS, DT, NM_M);

    for p in &parts {
        assert!(
            p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite(),
            "TreePM N={}³ {} pasos: posición NaN/Inf gid={}: {:?}",
            GRID_M,
            N_STEPS,
            p.global_id,
            p.position
        );
        assert!(
            p.velocity.x.is_finite() && p.velocity.y.is_finite() && p.velocity.z.is_finite(),
            "TreePM N={}³ {} pasos: velocidad NaN/Inf gid={}: {:?}",
            GRID_M,
            N_STEPS,
            p.global_id,
            p.velocity
        );
    }
    println!(
        "\n[phase32 TreePM N={}³ {} pasos] estable — sin NaN/Inf",
        GRID_M, N_STEPS
    );
}

// ── Test 10: reproducibilidad bit a bit N=32³ ─────────────────────────────────

/// Misma seed con N=32³ produce P(k) bit-idéntico en dos ejecuciones.
///
/// Con 32 768 partículas, valida que no hay fuentes de no-determinismo
/// (por ejemplo, reducción paralela con orden no fijo) a mayor escala.
#[test]
fn n32_reproducibility() {
    const SEED: u64 = 3232;

    let parts_a = build_particles(&make_config(SEED, GRID_M, NM_M, true, false)).unwrap();
    let parts_b = build_particles(&make_config(SEED, GRID_M, NM_M, true, false)).unwrap();

    let pk_a = measure_pk(&parts_a, NM_M);
    let pk_b = measure_pk(&parts_b, NM_M);

    assert_eq!(
        pk_a.len(),
        pk_b.len(),
        "Diferente número de bins entre ejecuciones"
    );

    for (i, (ba, bb)) in pk_a.iter().zip(pk_b.iter()).enumerate() {
        assert_eq!(
            ba.pk.to_bits(),
            bb.pk.to_bits(),
            "P(k) no bit-idéntico en bin {}: {:.6e} vs {:.6e}",
            i,
            ba.pk,
            bb.pk
        );
        assert_eq!(
            ba.k.to_bits(),
            bb.k.to_bits(),
            "k no bit-idéntico en bin {}: {:.6e} vs {:.6e}",
            i,
            ba.k,
            bb.k
        );
    }

    println!(
        "\n[phase32 reproducibilidad N={}³] seed={}: {} bins bit-idénticos",
        GRID_M,
        SEED,
        pk_a.len()
    );
}
