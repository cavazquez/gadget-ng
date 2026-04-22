//! Tests de validación para función de transferencia Eisenstein–Hu y normalización σ₈ — Fase 27.
//!
//! ## Cobertura
//!
//! 1. `eh_transfer_low_k_is_one`:
//!    T(k=1e-4 h/Mpc) ≈ 1 — límite k→0 de la función de transferencia.
//!
//! 2. `eh_transfer_high_k_suppressed`:
//!    T(k=10 h/Mpc) << T(k=0.1 h/Mpc) — supresión correcta a alto k.
//!
//! 3. `sigma8_normalization_matches_target`:
//!    La amplitud calculada para σ₈=0.8 es autoconsistente (dentro del 0.1%).
//!
//! 4. `eh_spectrum_differs_from_power_law`:
//!    T²(k=1 h/Mpc) << 1 → el espectro EH difiere fuertemente del power-law a alto k.
//!
//! 5. `legacy_amplitude_still_works`:
//!    `IcKind::Zeldovich` con `transfer = PowerLaw` sigue compilando y generando partículas.
//!
//! 6. `positions_in_box_with_eh`:
//!    Posiciones en `[0, box_size)` con ICs Eisenstein–Hu.
//!
//! 7. `pm_run_stable_with_eh_ics`:
//!    10 pasos de PM con ICs EH no produce NaN/Inf.
//!
//! 8. `treepm_run_stable_with_eh_ics`:
//!    10 pasos de TreePM con ICs EH no produce NaN/Inf.

use gadget_ng_analysis::power_spectrum::power_spectrum;
use gadget_ng_core::{
    amplitude_for_sigma8, build_particles, cosmology::CosmologyParams, sigma_sq_unit,
    transfer_eh_nowiggle, wrap_position, CosmologySection, EisensteinHuParams, GravitySection,
    GravitySolver, IcKind, InitialConditionsSection, OutputSection, PerformanceSection, RunConfig,
    SimulationSection, TimestepSection, TransferKind, UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_pm::PmSolver;
use gadget_ng_treepm::TreePmSolver;

// ── Constantes ────────────────────────────────────────────────────────────────

const G: f64 = 1.0;
const BOX: f64 = 1.0;
const GRID: usize = 8;
const N_PART: usize = 512; // 8³
const NM: usize = 8;

const OMEGA_M: f64 = 0.315;
const OMEGA_B: f64 = 0.049;
const H_DIMLESS: f64 = 0.674;
const T_CMB: f64 = 2.7255;
const N_S: f64 = 0.965;
const SIGMA8_TARGET: f64 = 0.8;
const BOX_MPC_H: f64 = 100.0;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn planck18_params() -> EisensteinHuParams {
    EisensteinHuParams {
        omega_m: OMEGA_M,
        omega_b: OMEGA_B,
        h: H_DIMLESS,
        t_cmb: T_CMB,
    }
}

/// Configuración ΛCDM con ICs Eisenstein–Hu y normalización σ₈.
fn eh_config(seed: u64) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.002,
            num_steps: 10,
            softening: 0.02,
            gravitational_constant: G,
            particle_count: N_PART,
            box_size: BOX,
            seed,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Zeldovich {
                seed,
                grid_size: GRID,
                spectral_index: N_S,
                amplitude: 1.0e-4, // fallback (no usado cuando sigma8 = Some)
                transfer: TransferKind::EisensteinHu,
                sigma8: Some(SIGMA8_TARGET),
                omega_b: OMEGA_B,
                h: H_DIMLESS,
                t_cmb: T_CMB,
                box_size_mpc_h: Some(BOX_MPC_H),
                use_2lpt: false,
                normalization_mode: gadget_ng_core::NormalizationMode::Legacy,
            },
        },
        output: OutputSection::default(),
        gravity: GravitySection {
            solver: gadget_ng_core::SolverKind::Pm,
            pm_grid_size: NM,
            ..GravitySection::default()
        },
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection {
            enabled: true,
            periodic: true,
            omega_m: OMEGA_M,
            omega_lambda: 0.685,
            h0: 0.1,
            a_init: 0.02,
                auto_g: false,
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
    }
}

/// Configuración legacy de Fase 26 (ley de potencia pura).
fn legacy_config(seed: u64) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.002,
            num_steps: 10,
            softening: 0.02,
            gravitational_constant: G,
            particle_count: N_PART,
            box_size: BOX,
            seed,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Zeldovich {
                seed,
                grid_size: GRID,
                spectral_index: -2.0,
                amplitude: 1.0e-4,
                transfer: TransferKind::PowerLaw,
                sigma8: None,
                omega_b: OMEGA_B,
                h: H_DIMLESS,
                t_cmb: T_CMB,
                box_size_mpc_h: None,
                use_2lpt: false,
                normalization_mode: gadget_ng_core::NormalizationMode::Legacy,
            },
        },
        output: OutputSection::default(),
        gravity: GravitySection {
            solver: gadget_ng_core::SolverKind::Pm,
            pm_grid_size: NM,
            ..GravitySection::default()
        },
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection {
            enabled: true,
            periodic: true,
            omega_m: 1.0,
            omega_lambda: 0.0,
            h0: 0.1,
            a_init: 0.02,
                auto_g: false,
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
    }
}

// ── Test 1: T(k→0) = 1 ───────────────────────────────────────────────────────

/// La función de transferencia debe ser ≈ 1 para k→0 (escalas super-horizonte).
#[test]
fn eh_transfer_low_k_is_one() {
    let p = planck18_params();
    let t = transfer_eh_nowiggle(1e-4, &p);
    assert!(
        (t - 1.0).abs() < 1e-3,
        "T(k=1e-4 h/Mpc) = {:.6} ≠ 1 (tolerancia 1e-3)",
        t
    );
}

// ── Test 2: supresión en alto k ───────────────────────────────────────────────

/// T(k=10 h/Mpc) debe estar fuertemente suprimido respecto a T(k=0.1 h/Mpc).
///
/// Para Planck18 con EH no-wiggle:
/// - T(k=0.1 h/Mpc) ≈ 0.14 (k > k_eq ≈ 0.015 h/Mpc)
/// - T(k=10 h/Mpc) ≈ 1e-4 (muy suprimido)
#[test]
fn eh_transfer_high_k_suppressed() {
    let p = planck18_params();
    let t_intermediate = transfer_eh_nowiggle(0.1, &p);
    let t_high = transfer_eh_nowiggle(10.0, &p);

    assert!(
        t_high < t_intermediate,
        "T(10) = {:.6} ≥ T(0.1) = {:.6} — la función de transferencia no suprime alto k",
        t_high, t_intermediate
    );

    assert!(
        t_high < 0.01,
        "T(10 h/Mpc) = {:.6} no está lo suficientemente suprimido (esperado < 0.01 para Planck18)",
        t_high
    );
}

// ── Test 3: σ₈ normalización autoconsistente ──────────────────────────────────

/// La amplitud A calculada para σ₈=0.8 reproduce el valor exacto cuando se rehace
/// la integral σ²(8) × A² = σ₈_target².
///
/// Esta es la prueba de autoconsistencia de la normalización analítica.
/// La verificación empírica completa (usando P(k) medido del campo de partículas
/// con N ≥ 128³) está en los scripts Python de experimentos.
#[test]
fn sigma8_normalization_matches_target() {
    let p = planck18_params();
    let sigma8_target = SIGMA8_TARGET;

    let amp = amplitude_for_sigma8(sigma8_target, N_S, &p);
    let s2_unit = sigma_sq_unit(8.0, N_S, &p);
    let sigma8_recovered = amp * s2_unit.sqrt();

    let rel_err = (sigma8_recovered - sigma8_target).abs() / sigma8_target;
    assert!(
        rel_err < 1e-3,
        "σ₈ recuperado = {:.6} vs target = {:.6} (error relativo = {:.4}%)",
        sigma8_recovered, sigma8_target, rel_err * 100.0
    );
}

// ── Test 4: EH difiere de ley de potencia ─────────────────────────────────────

/// A k = 1.0 h/Mpc, T²(k) << 1, lo que demuestra que el espectro EH difiere
/// significativamente del espectro de ley de potencia pura (que tiene T = 1).
///
/// Adicionalmente verificamos que las ICs EH y legacy producen posiciones distintas
/// (para la misma seed), lo que confirma que el código usa efectivamente T(k) != 1.
#[test]
fn eh_spectrum_differs_from_power_law() {
    let p = planck18_params();

    // A k = 1 h/Mpc (bien en el régimen suprimido), T²(k) debe ser << 1.
    let t_sq = transfer_eh_nowiggle(1.0, &p).powi(2);
    assert!(
        t_sq < 0.1,
        "T²(k=1 h/Mpc) = {:.4} — no hay supresión suficiente (esperado < 0.1)",
        t_sq
    );

    // Con misma seed, las ICs EH y legacy deben diferir (T(k) != 1 cambia el campo).
    let cfg_eh = eh_config(77);
    let cfg_pl = legacy_config(77);

    let parts_eh = build_particles(&cfg_eh).expect("IC EH build");
    let parts_pl = build_particles(&cfg_pl).expect("IC PL build");

    // Calcular P(k) con la API correcta.
    let pos_eh: Vec<Vec3> = parts_eh.iter().map(|p| p.position).collect();
    let m_eh: Vec<f64> = parts_eh.iter().map(|p| p.mass).collect();
    let pos_pl: Vec<Vec3> = parts_pl.iter().map(|p| p.position).collect();
    let m_pl: Vec<f64> = parts_pl.iter().map(|p| p.mass).collect();

    let pk_eh = power_spectrum(&pos_eh, &m_eh, BOX, GRID);
    let pk_pl = power_spectrum(&pos_pl, &m_pl, BOX, GRID);

    // Al menos un bin de P(k) debe diferir entre los dos casos.
    let any_different = pk_eh.iter().zip(pk_pl.iter()).any(|(b_eh, b_pl)| {
        b_eh.pk.is_finite() && b_pl.pk.is_finite() && b_eh.pk > 0.0 && b_pl.pk > 0.0
            && (b_eh.pk - b_pl.pk).abs() / (b_eh.pk + b_pl.pk) > 0.01
    });
    assert!(
        any_different,
        "P(k) EH y P(k) power-law son idénticos — la función de transferencia no tiene efecto"
    );
}

// ── Test 5: retrocompatibilidad legacy ────────────────────────────────────────

/// Las ICs con `transfer = PowerLaw` (comportamiento de Fase 26) siguen funcionando.
/// Todas las posiciones deben estar en [0, box_size).
#[test]
fn legacy_amplitude_still_works() {
    let cfg = legacy_config(123);
    let parts = build_particles(&cfg).expect("IC legacy build debería funcionar");

    assert_eq!(parts.len(), N_PART, "Número incorrecto de partículas");

    for p in &parts {
        assert!(p.position.x >= 0.0 && p.position.x < BOX,
            "x = {} fuera de [0, {})", p.position.x, BOX);
        assert!(p.position.y >= 0.0 && p.position.y < BOX,
            "y = {} fuera de [0, {})", p.position.y, BOX);
        assert!(p.position.z >= 0.0 && p.position.z < BOX,
            "z = {} fuera de [0, {})", p.position.z, BOX);
    }
}

// ── Test 6: posiciones en caja con ICs EH ────────────────────────────────────

/// Todas las posiciones con ICs Eisenstein–Hu deben estar en `[0, box_size)`.
#[test]
fn positions_in_box_with_eh() {
    let cfg = eh_config(55);
    let parts = build_particles(&cfg).expect("IC EH build");

    assert_eq!(parts.len(), N_PART);

    for p in &parts {
        assert!(
            p.position.x >= 0.0 && p.position.x < BOX,
            "x = {} fuera de [0, {})", p.position.x, BOX
        );
        assert!(
            p.position.y >= 0.0 && p.position.y < BOX,
            "y = {} fuera de [0, {})", p.position.y, BOX
        );
        assert!(
            p.position.z >= 0.0 && p.position.z < BOX,
            "z = {} fuera de [0, {})", p.position.z, BOX
        );
    }
}

// ── Test 7: PM estable con ICs EH ────────────────────────────────────────────

/// 10 pasos de leapfrog cosmológico con PM y ICs Eisenstein–Hu no produce NaN/Inf.
#[test]
fn pm_run_stable_with_eh_ics() {
    let cfg = eh_config(888);
    let mut parts = build_particles(&cfg).expect("IC EH build");
    let cosmo = CosmologyParams::new(OMEGA_M, 0.685, 0.1);
    let mut a = cfg.cosmology.a_init;
    let dt = cfg.simulation.dt;

    let pm = PmSolver { grid_size: NM, box_size: BOX };
    let mut scratch = vec![Vec3::zero(); N_PART];

    for _ in 0..10 {
        let g_cosmo = G / a;
        let (drift, kick_half, kick_half2) = cosmo.drift_kick_factors(a, dt);
        let cf = CosmoFactors { drift, kick_half, kick_half2 };
        a = cosmo.advance_a(a, dt);

        leapfrog_cosmo_kdk_step(&mut parts, cf, &mut scratch, |ps, acc| {
            let pos: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let m: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let idx: Vec<usize> = (0..ps.len()).collect();
            pm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, acc);
        });

        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
    }

    for p in &parts {
        assert!(
            p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite(),
            "Posición NaN/Inf en PM+EH en gid={}: {:?}", p.global_id, p.position
        );
        assert!(
            p.velocity.x.is_finite() && p.velocity.y.is_finite() && p.velocity.z.is_finite(),
            "Velocidad NaN/Inf en PM+EH en gid={}: {:?}", p.global_id, p.velocity
        );
    }
}

// ── Test 8: TreePM estable con ICs EH ────────────────────────────────────────

/// 10 pasos de leapfrog cosmológico con TreePM y ICs Eisenstein–Hu no produce NaN/Inf.
#[test]
fn treepm_run_stable_with_eh_ics() {
    let cfg = {
        let mut c = eh_config(999);
        c.gravity.solver = gadget_ng_core::SolverKind::TreePm;
        c.gravity.theta = 0.5;
        c
    };

    let mut parts = build_particles(&cfg).expect("IC EH build");
    let cosmo = CosmologyParams::new(OMEGA_M, 0.685, 0.1);
    let mut a = cfg.cosmology.a_init;
    let dt = cfg.simulation.dt;

    let treepm = TreePmSolver {
        grid_size: NM,
        box_size: BOX,
        r_split: 0.0,
    };
    let mut scratch = vec![Vec3::zero(); N_PART];

    for _ in 0..10 {
        let g_cosmo = G / a;
        let (drift, kick_half, kick_half2) = cosmo.drift_kick_factors(a, dt);
        let cf = CosmoFactors { drift, kick_half, kick_half2 };
        a = cosmo.advance_a(a, dt);

        leapfrog_cosmo_kdk_step(&mut parts, cf, &mut scratch, |ps, acc| {
            let pos: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let m: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let idx: Vec<usize> = (0..ps.len()).collect();
            treepm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, acc);
        });

        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
    }

    for p in &parts {
        assert!(
            p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite(),
            "Posición NaN/Inf en TreePM+EH en gid={}: {:?}", p.global_id, p.position
        );
        assert!(
            p.velocity.x.is_finite() && p.velocity.y.is_finite() && p.velocity.z.is_finite(),
            "Velocidad NaN/Inf en TreePM+EH en gid={}: {:?}", p.global_id, p.velocity
        );
    }
}
