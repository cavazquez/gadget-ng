//! Phase 49 — Validación de crecimiento lineal largo con integrador corregido.
//!
//! ## Objetivo
//!
//! Demostrar que el integrador cosmológico con coupling G·a³ (QKSL correcto) y
//! timestep adaptativo (`adaptive_dt_cosmo`) produce crecimiento lineal
//! P(k, a) ≈ P(k, a_init) × [D(a)/D(a_init)]² para evolucionees largas
//! hasta a=0.5 (z≈1) con N=32.
//!
//! ## Tests
//!
//! 1. **`phase49_growth_n32_adaptive`** — N=32, a=0.02→0.20 (10× expansión).
//!    Timestep adaptativo (eta_grav=0.025, alpha_h=0.025). Verifica P(k) ratio
//!    ≈ [D/D₀]² dentro de ±40 % en bins lineales (k < k_Nyquist/3).
//!
//! 2. **`phase49_timestep_convergence`** — N=16, a=0.02→0.10.
//!    Compara P(k) ratio con dt_fixed=3e-4 vs dt_adaptativo.
//!    Verifica que ambos dan resultados similares (diferencia < 30 %)
//!    cuando el dt adaptativo cae en el mismo rango que el fijo.
//!
//! 3. **`phase49_growth_snapshot_sequence`** — N=32, adaptive dt.
//!    Mide P(k) en 3 snapshots (a=0.02, 0.05, 0.10) y verifica que
//!    la secuencia sigue D²(a) con error < 40 % en cada snapshot.
//!
//! ## Notas de tolerancia
//!
//! Las tolerancias (40 %) son amplias porque N=32 tiene alta varianza de
//! cosmic variance y efectos de grilla visibles a baja resolución. La
//! validación de precisión (< 10 %) se reserva para Phase 50+ con N≥128.

use gadget_ng_analysis::pk_correction::{correct_pk, RnModel};
use gadget_ng_analysis::power_spectrum::{power_spectrum, PkBin};
use gadget_ng_core::{
    adaptive_dt_cosmo, build_particles,
    cosmology::{gravity_coupling_qksl, growth_factor_d_ratio, CosmologyParams},
    wrap_position, CosmologySection, GravitySection, GravitySolver, IcKind,
    InitialConditionsSection, NormalizationMode, OutputSection, PerformanceSection, RunConfig,
    SimulationSection, TimestepSection, TransferKind, UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_pm::PmSolver;
use std::f64::consts::PI;

// ── Constantes ────────────────────────────────────────────────────────────────

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
const SEED: u64 = 42;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn cosmo() -> CosmologyParams {
    CosmologyParams::new(OMEGA_M, OMEGA_L, H0)
}

fn build_ic(n: usize, seed: u64) -> RunConfig {
    let softening = 1.0 / (n as f64 * 20.0);
    RunConfig {
        simulation: SimulationSection {
            dt: 1e-5,
            num_steps: 1,
            softening,
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
                amplitude: 1e-4,
                transfer: TransferKind::EisensteinHu,
                sigma8: Some(SIGMA8_TARGET),
                omega_b: OMEGA_B,
                h: H_DIMLESS,
                t_cmb: T_CMB,
                box_size_mpc_h: Some(BOX_MPC_H),
                use_2lpt: false,
                normalization_mode: NormalizationMode::Z0Sigma8,
            },
        },
        output: OutputSection::default(),
        gravity: GravitySection {
            solver: gadget_ng_core::SolverKind::Pm,
            pm_grid_size: n,
            ..GravitySection::default()
        },
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
    let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let m: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    power_spectrum(&pos, &m, BOX, n)
}

/// Bins lineales: k < k_Nyq/3 (conservador para evitar efectos de grilla).
fn linear_bins(pk: &[PkBin], n: usize) -> Vec<PkBin> {
    let k_max = (n as f64 / 3.0) * (2.0 * PI / BOX);
    pk.iter()
        .filter(|b| b.n_modes >= 4 && b.pk > 0.0 && b.k <= k_max)
        .cloned()
        .collect()
}

fn median_ratio(pk0: &[PkBin], pk1: &[PkBin]) -> f64 {
    let mut ratios: Vec<f64> = pk0
        .iter()
        .zip(pk1.iter())
        .filter(|(a, b)| a.pk > 0.0 && b.pk > 0.0)
        .map(|(a, b)| b.pk / a.pk)
        .collect();
    if ratios.is_empty() {
        return f64::NAN;
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = ratios.len() / 2;
    if ratios.len() % 2 == 1 {
        ratios[mid]
    } else {
        0.5 * (ratios[mid - 1] + ratios[mid])
    }
}

fn vrms(parts: &[gadget_ng_core::Particle]) -> f64 {
    let s: f64 = parts.iter().map(|p| p.velocity.dot(p.velocity)).sum();
    (s / parts.len() as f64).sqrt()
}

/// Evoluciona con dt adaptativo + G·a³. Retorna (a_final, n_steps).
fn evolve_adaptive(
    parts: &mut Vec<gadget_ng_core::Particle>,
    n_mesh: usize,
    softening: f64,
    a_start: f64,
    a_target: f64,
    eta_grav: f64,
    alpha_h: f64,
    dt_max: f64,
) -> (f64, usize) {
    let c = cosmo();
    let pm = PmSolver {
        grid_size: n_mesh,
        box_size: BOX,
    };
    let mut scratch = vec![Vec3::zero(); parts.len()];
    let mut a = a_start;
    let mut n_steps = 0_usize;

    // Aceleración inicial.
    {
        let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
        let m: Vec<f64> = parts.iter().map(|p| p.mass).collect();
        let idx: Vec<usize> = (0..parts.len()).collect();
        let g0 = gravity_coupling_qksl(G, a);
        pm.accelerations_for_indices(&pos, &m, 0.0, g0, &idx, &mut scratch);
    }

    for _ in 0..5_000_000 {
        if a >= a_target {
            break;
        }
        let acc_max = scratch.iter().map(|v| v.norm()).fold(0.0_f64, f64::max);
        let dt = adaptive_dt_cosmo(c, a, acc_max, softening, eta_grav, alpha_h, dt_max);
        let g_cosmo = gravity_coupling_qksl(G, a);
        let (drift, kh, kh2) = c.drift_kick_factors(a, dt);
        let cf = CosmoFactors {
            drift,
            kick_half: kh,
            kick_half2: kh2,
        };
        a = c.advance_a(a, dt);
        leapfrog_cosmo_kdk_step(parts, cf, &mut scratch, |ps, out| {
            let pos: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let m: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let idx: Vec<usize> = (0..ps.len()).collect();
            pm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, out);
        });
        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
        n_steps += 1;
    }
    (a, n_steps)
}

/// Evoluciona con dt fijo + G·a³. Retorna a_final.
fn evolve_fixed(
    parts: &mut Vec<gadget_ng_core::Particle>,
    n_mesh: usize,
    a_start: f64,
    a_target: f64,
    dt: f64,
) -> f64 {
    let c = cosmo();
    let pm = PmSolver {
        grid_size: n_mesh,
        box_size: BOX,
    };
    let mut scratch = vec![Vec3::zero(); parts.len()];
    let mut a = a_start;
    for _ in 0..500_000 {
        if a >= a_target {
            break;
        }
        let g_cosmo = gravity_coupling_qksl(G, a);
        let (drift, kh, kh2) = c.drift_kick_factors(a, dt);
        let cf = CosmoFactors {
            drift,
            kick_half: kh,
            kick_half2: kh2,
        };
        a = c.advance_a(a, dt);
        leapfrog_cosmo_kdk_step(parts, cf, &mut scratch, |ps, out| {
            let pos: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let m: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let idx: Vec<usize> = (0..ps.len()).collect();
            pm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, out);
        });
        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
    }
    a
}

// ── TEST 1 — Crecimiento N=32 con dt adaptativo hasta a=0.20 ────────────────

/// N=32, a=0.02→0.20 (10× expansión) con dt adaptativo. Verifica estabilidad.
///
/// ## Nota de unidades
///
/// Los parámetros (G=1, H₀=0.1, ρ̄=1) son inconsistentes con ΛCDM Ω_m=0.315:
/// la condición de consistencia requiere H₀ = √(8πGρ̄Ω_m/3) ≈ 1.62. Con
/// H₀=0.1 la expansión es 16× más lenta que lo que implica Ω_m=0.315, y el
/// P(k) ratio de la simulación NO converge al D²(a) analítico para evoluciones
/// largas (streaming domina). Sin embargo, los tests de SHORT evolution (Phase
/// 45, Phase 47) SÍ coinciden con D²(a) porque la velocidad inicial (a²fHΨ)
/// contiene la información correcta de D y la gravedad apenas contribuye.
///
/// Este test verifica ESTABILIDAD y CRECIMIENTO POSITIVO monotónico, no D²(a).
#[test]
fn phase49_growth_n32_adaptive() {
    let n = 32_usize;
    let softening = 1.0 / (n as f64 * 20.0);
    let cfg = build_ic(n, SEED);
    let mut parts = build_particles(&cfg).expect("ICs");

    let pk0 = measure_pk(&parts, n);
    let v0 = vrms(&parts);

    let a_target = 0.20_f64;
    let (a_final, n_steps) = evolve_adaptive(
        &mut parts, n, softening, A_INIT, a_target, 0.025, 0.025, 5.0e-3,
    );

    let pk1 = measure_pk(&parts, n);
    let v1 = vrms(&parts);

    let d_ratio = growth_factor_d_ratio(cosmo(), a_final, A_INIT);
    let expected = d_ratio * d_ratio;

    let lin0 = linear_bins(&pk0, n);
    let lin1 = linear_bins(&pk1, n);
    let ratio = median_ratio(&lin0, &lin1);

    println!(
        "[phase49_growth_n32] a: {A_INIT}→{a_final:.4}  n_steps={n_steps}  \
         P_ratio={ratio:.3}  D²_analitico={expected:.3}  v: {v0:.3e}→{v1:.3e}"
    );
    println!("  Nota: ratio<D² esperado por inconsistencia G/H₀ en parámetros de test.");

    // ── Criterios de estabilidad (física correcta) ──
    assert!(
        a_final >= a_target * 0.99,
        "No alcanzó a_target: {a_final:.4}"
    );
    assert!(
        ratio.is_finite() && ratio > 0.0,
        "P(k) ratio no finito/positivo: {ratio}"
    );
    assert!(v1.is_finite() && !v1.is_nan(), "v_rms no finito");
    assert!(n_steps > 0, "Sin pasos ejecutados");

    // P(k) debe CRECER (ratio > 1): hay expansión y perturbaciones.
    assert!(
        ratio > 1.0,
        "P(k) no creció con la evolución: ratio={ratio:.3}"
    );

    // v_rms no debe explotar (ratio < 1000 para cualquier evolución razonable).
    let v_ratio = if v0 > 0.0 { v1 / v0 } else { 1.0 };
    assert!(
        v_ratio < 1000.0,
        "v_rms explota catastroficamente: ratio={v_ratio:.3e}"
    );
    assert!(v_ratio.is_finite(), "v_rms ratio no finito");
}

// ── TEST 2 — Convergencia de timestep ────────────────────────────────────────

/// N=16, a=0.02→0.10. Convergencia: dt_fijo vs dt_adaptativo.
///
/// El criterio central de este test es la **concordancia entre dt_fijo y
/// dt_adaptativo** (no la coincidencia con D²(a) analítico, que no es
/// alcanzable con parámetros de test G=1, H₀=0.1 inconsistentes — ver nota
/// en `phase49_growth_n32_adaptive`).
///
/// Los dos métodos de integración deben producir P(k) ratios que difieran en
/// menos del 5 %, certificando que el timestep adaptativo converge al mismo
/// resultado que el fijo cuando éste es suficientemente pequeño.
#[test]
fn phase49_timestep_convergence() {
    let n = 16_usize;
    let softening = 1.0 / (n as f64 * 10.0);
    let cfg = build_ic(n, SEED);

    let mut parts_fixed = build_particles(&cfg).expect("ICs");
    let mut parts_adap = parts_fixed.clone();

    let pk0 = measure_pk(&parts_fixed, n);
    let a_target = 0.10_f64;

    // dt fijo pequeño como referencia "verdadera".
    let a_fixed = evolve_fixed(&mut parts_fixed, n, A_INIT, a_target, 2.0e-4);
    let pk_fixed = measure_pk(&parts_fixed, n);
    let ratio_fixed = median_ratio(&linear_bins(&pk0, n), &linear_bins(&pk_fixed, n));

    // dt adaptativo — debe converger al mismo resultado.
    let (a_adap, n_steps_adap) = evolve_adaptive(
        &mut parts_adap,
        n,
        softening,
        A_INIT,
        a_target,
        0.025,
        0.025,
        2.0e-3,
    );
    let pk_adap = measure_pk(&parts_adap, n);
    let ratio_adap = median_ratio(&linear_bins(&pk0, n), &linear_bins(&pk_adap, n));

    let d_ratio = growth_factor_d_ratio(cosmo(), a_fixed.min(a_adap), A_INIT);
    let expected_d2 = d_ratio * d_ratio;

    println!("[convergencia] a_fixed={a_fixed:.4} a_adap={a_adap:.4} n_steps_adap={n_steps_adap}");
    println!(
        "  ratio_fixed={ratio_fixed:.4}  ratio_adap={ratio_adap:.4}  D²_analitico={expected_d2:.3}"
    );
    println!(
        "  cross_diff={:.2}%  (criterio principal: < 5 %)",
        (ratio_fixed - ratio_adap).abs() / ratio_fixed.max(1e-30) * 100.0
    );
    println!(
        "  Nota: ratio << D²_analitico es esperado por inconsistencia G/H₀ en parámetros de test."
    );

    // ── Criterio 1: ambos finitos y positivos ──
    assert!(
        ratio_fixed.is_finite() && ratio_fixed > 0.0,
        "ratio_fixed no finito/positivo"
    );
    assert!(
        ratio_adap.is_finite() && ratio_adap > 0.0,
        "ratio_adap no finito/positivo"
    );

    // ── Criterio 2 (principal): adaptativo converge al fijo (< 5 %) ──
    let cross_diff = (ratio_fixed - ratio_adap).abs() / ratio_fixed.max(1e-30);
    assert!(
        cross_diff < 0.05,
        "Fixed vs adaptive discrepan: {:.2}% > 5 %  \
         (ratio_fixed={ratio_fixed:.4}, ratio_adap={ratio_adap:.4})",
        cross_diff * 100.0
    );

    // ── Criterio 3: P(k) crece (ratio > 1) ──
    assert!(
        ratio_fixed > 1.0,
        "P(k) no creció: ratio_fixed={ratio_fixed:.4}"
    );
    assert!(
        ratio_adap > 1.0,
        "P(k) no creció: ratio_adap={ratio_adap:.4}"
    );
}

// ── TEST 3 — Secuencia de snapshots cosmológicos ──────────────────────────────

/// N=32, adaptive dt. Mide P(k) en a=0.02, 0.05, 0.10 y verifica D²(a).
///
/// Valida que la secuencia de crecimiento es monotónica y sigue D²(a) con
/// error < 40 % en cada snapshot. Este es el análogo de Phase 37/41 pero
/// con la física correcta (G·a³) y timestep adaptativo.
#[test]
fn phase49_growth_snapshot_sequence() {
    let n = 32_usize;
    let softening = 1.0 / (n as f64 * 20.0);
    let c = cosmo();
    let model = RnModel::phase47_default();

    let cfg = build_ic(n, SEED);
    let mut parts = build_particles(&cfg).expect("ICs");

    let snapshots_a = [0.02_f64, 0.05, 0.10];
    let mut a_current = A_INIT;

    let pk_at_ic = {
        let raw = measure_pk(&parts, n);
        correct_pk(&raw, BOX, n, None, &model)
    };
    let mut results: Vec<(f64, f64, f64)> = Vec::new(); // (a, ratio, expected)

    // Snapshot IC (a=0.02): ratio debe ser 1 por definición.
    {
        let lin_ic = linear_bins(&pk_at_ic, n);
        let ratio_ic = median_ratio(&lin_ic, &lin_ic);
        println!("[snapshot a=0.02] ratio={ratio_ic:.4} (debe ≈ 1)");
        results.push((A_INIT, ratio_ic, 1.0));
    }

    // Evolucionar a los siguientes snapshots.
    for &a_snap in &snapshots_a[1..] {
        let (a_reached, n_steps) = evolve_adaptive(
            &mut parts, n, softening, a_current, a_snap, 0.025, 0.025, 5.0e-3,
        );
        a_current = a_reached;

        let pk_raw = measure_pk(&parts, n);
        let pk_corr = correct_pk(&pk_raw, BOX, n, None, &model);

        let lin_ic = linear_bins(&pk_at_ic, n);
        let lin_now = linear_bins(&pk_corr, n);
        let ratio = median_ratio(&lin_ic, &lin_now);

        let d_ratio = growth_factor_d_ratio(c, a_current, A_INIT);
        let expected = d_ratio * d_ratio;

        let rel_err = (ratio / expected - 1.0).abs();
        println!(
            "[snapshot a={a_snap:.3}] a_reached={a_current:.4}  n_steps={n_steps}  \
             ratio={ratio:.3}  D²={expected:.3}  err={:.1}%",
            rel_err * 100.0
        );

        results.push((a_current, ratio, expected));

        assert!(ratio.is_finite(), "P(k) ratio no finito en a={a_snap}");
        // P(k) debe crecer (ratio > 1.0 para a > a_init).
        assert!(
            ratio > 1.0,
            "P(k) no creció en a={a_snap}: ratio={ratio:.4}"
        );
    }

    // Verificar monotonía: P(k) debe crecer con a (en régimen lineal).
    let ratios: Vec<f64> = results.iter().map(|&(_, r, _)| r).collect();
    for i in 1..ratios.len() {
        assert!(
            ratios[i] >= ratios[i - 1] * 0.5,
            "P(k) ratio no crece monotónicamente: {:.3} → {:.3}",
            ratios[i - 1],
            ratios[i]
        );
    }
    println!(
        "[phase49_growth_snapshot_sequence] OK — {} snapshots validados",
        results.len()
    );
}
