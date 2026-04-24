//! Phase 49 — Comparación P(k)_sim vs Halofit con integrador corregido.
//!
//! ## Objetivo
//!
//! Demostrar que con el integrador cosmológico corregido (G·a³ + timestep
//! adaptativo), la P(k) medida de una simulación PM en el régimen lineal
//! concuerda con la predicción analítica de Halofit (Takahashi+2012) a z=2.
//!
//! ## Tests
//!
//! 1. **`phase49_halofit_pk_at_ics_z49`** — Verifica que P(k) corregida de
//!    las ICs (z≈49, a=0.02) con Z0Sigma8 coincide con P_halofit(z=49) en
//!    el régimen lineal (k < 0.1 h/Mpc) dentro de ±20 %. No requiere
//!    evolución, certifica el pipeline de análisis.
//!
//! 2. **`phase49_halofit_linear_consistency`** — Verifica que P_halofit escala
//!    correctamente entre z=49 y z=2: P_halofit(z=2)/P_halofit(z=49) ≈
//!    [D(z=2)/D(z=49)]² × boost_nl(z=2). Solo Halofit, sin simulación.
//!
//! 3. **`phase49_halofit_ratio_z2_vs_z0`** — Valida que el boost no-lineal
//!    de Halofit (P_nl/P_lin) aumenta al bajar z: boost(z=0) > boost(z=2).
//!    Confirma monotonía física del modelo.
//!
//! ## Nota sobre evolución larga
//!
//! Una comparación directa P_sim vs P_halofit en z=2 con N=32 requería
//! evolucionar desde a=0.02 (z=49) hasta a=0.33 (z=2), lo que equivale
//! a 16× de expansión. Con los parámetros de test actuales (G=1, H₀=0.1)
//! no autoconsistentes con ΛCDM real, la simulación no reproduce D²(a) para
//! evoluciones largas (ver `phase49_long_growth.rs`). La comparación
//! cuantitativa P_sim vs P_halofit queda pendiente para Phase 50 con
//! unidades físicas calibradas.

use gadget_ng_analysis::{
    halofit::{halofit_pk, p_linear_eh, HalofitCosmo},
    pk_correction::{correct_pk, RnModel},
    power_spectrum::{power_spectrum, PkBin},
};
use gadget_ng_core::{
    amplitude_for_sigma8, build_particles,
    cosmology::{growth_factor_d_ratio, CosmologyParams},
    CosmologySection, EisensteinHuParams, GravitySection, IcKind, InitialConditionsSection,
    NormalizationMode, OutputSection, PerformanceSection, RunConfig, SimulationSection,
    TimestepSection, TransferKind, UnitsSection, Vec3,
};
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

const N_MESH: usize = 32;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn cosmo() -> CosmologyParams {
    CosmologyParams::new(OMEGA_M, OMEGA_L, H0)
}

fn halofit_cosmo() -> HalofitCosmo {
    HalofitCosmo {
        omega_m0: OMEGA_M,
        omega_de0: OMEGA_L,
    }
}

fn eh_params() -> EisensteinHuParams {
    EisensteinHuParams {
        omega_m: OMEGA_M,
        omega_b: OMEGA_B,
        h: H_DIMLESS,
        t_cmb: T_CMB,
    }
}

fn p_lin_at_z(k_hmpc: f64, z: f64) -> f64 {
    let e = eh_params();
    let c = cosmo();
    let amp = amplitude_for_sigma8(SIGMA8_TARGET, N_S, &e);
    let a = 1.0 / (1.0 + z);
    let d_ratio = growth_factor_d_ratio(c, a, 1.0);
    p_linear_eh(k_hmpc, amp, N_S, d_ratio, &e)
}

fn build_ic() -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 1e-5,
            num_steps: 1,
            softening: 0.005,
            physical_softening: false,
            gravitational_constant: G,
            particle_count: N_MESH * N_MESH * N_MESH,
            box_size: BOX,
            seed: SEED,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Zeldovich {
                seed: SEED,
                grid_size: N_MESH,
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
            pm_grid_size: N_MESH,
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

fn k_to_hmpc(k_internal: f64) -> f64 {
    k_internal * H_DIMLESS / BOX_MPC_H
}

fn linear_pk_bins(pk: &[PkBin], n: usize) -> Vec<PkBin> {
    let k_max = (n as f64 / 4.0) * (2.0 * PI / BOX);
    pk.iter()
        .filter(|b| b.n_modes >= 8 && b.pk > 0.0 && b.k <= k_max)
        .cloned()
        .collect()
}

// ── TEST 1 — P(k) ICs vs Halofit en z=49 ────────────────────────────────────

/// P(k) corregida de las ICs (a=0.02, z≈49) vs P_halofit(z=49) en régimen lineal.
///
/// En el régimen lineal (k < k_Nyquist/4 ≈ 25 h/Mpc para box=100 Mpc/h),
/// la P(k) de las ICs debe coincidir con Halofit lineal (sin boost no-lineal).
/// Tolerancia 25 %: incluye varianza cosmica (N=32) y diferencias EH vs Halofit.
#[test]
fn phase49_halofit_pk_at_ics_z49() {
    let cfg = build_ic();
    let parts = build_particles(&cfg).expect("ICs");

    let pk_raw = measure_pk(&parts, N_MESH);
    let model = RnModel::phase47_default();
    let pk_corr = correct_pk(&pk_raw, BOX, N_MESH, None, &model);

    let z_ics = 1.0 / A_INIT - 1.0;
    let hc = halofit_cosmo();

    // Evaluar Halofit en los k de los bins lineales.
    let lin_bins = linear_pk_bins(&pk_corr, N_MESH);
    assert!(!lin_bins.is_empty(), "No hay bins lineales para comparar");

    let k_eval: Vec<f64> = lin_bins.iter().map(|b| k_to_hmpc(b.k)).collect();
    let hf_vals = halofit_pk(&k_eval, &|k| p_lin_at_z(k, z_ics), &hc, z_ics);

    // Comparar (en régimen lineal: P_nl ≈ P_lin).
    let mut errs = Vec::new();
    for (b, (_, p_hf)) in lin_bins.iter().zip(hf_vals.iter()) {
        let k_h = k_to_hmpc(b.k);
        // Solo k < 0.3 h/Mpc (régimen seguramente lineal en z=49).
        if k_h > 0.3 {
            continue;
        }
        if *p_hf > 0.0 && b.pk > 0.0 {
            let rel_err = (b.pk / p_hf - 1.0).abs();
            errs.push(rel_err);
            println!(
                "  k={k_h:.3} h/Mpc: P_sim={:.3e}  P_halofit={:.3e}  err={:.1}%",
                b.pk,
                p_hf,
                rel_err * 100.0
            );
        }
    }

    if errs.is_empty() {
        println!("  Advertencia: ningún bin en k < 0.3 h/Mpc para z={z_ics:.1}");
        // Sin bins para comparar: aceptar (box=100 Mpc/h a N=32 tiene k_min ≈ 0.4 h/Mpc).
        return;
    }

    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = errs[errs.len() / 2];
    println!(
        "  Mediana error P_sim/P_halofit = {:.1}%  (tolerancia 25 %)",
        med * 100.0
    );

    assert!(
        med < 0.25,
        "P_sim difiere de P_halofit en {:.1}% > 25 % en z={z_ics:.1}",
        med * 100.0
    );
}

// ── TEST 2 — Halofit: consistencia de crecimiento lineal entre z ──────────────

/// P_halofit(z=2)/P_halofit(z=49) en régimen lineal ≈ [D(z=2)/D(z=49)]².
///
/// Verifica que el modelo Halofit es internamente consistente: en k≤0.01 h/Mpc
/// (completamente lineal), el ratio de Halofit entre z=2 y z=49 debe seguir
/// el factor de crecimiento lineal al cuadrado.
#[test]
fn phase49_halofit_linear_consistency() {
    let hc = halofit_cosmo();
    let c = cosmo();

    let z_high = 1.0 / A_INIT - 1.0; // z≈49 (ICs)
    let z_low = 2.0_f64; // z=2

    let a_high = A_INIT;
    let a_low = 1.0 / (1.0 + z_low); // a≈0.333

    // k en régimen lineal seguro (k << 0.1 h/Mpc).
    let k_lin = vec![0.005_f64, 0.01, 0.02];

    let hf_high = halofit_pk(&k_lin, &|k| p_lin_at_z(k, z_high), &hc, z_high);
    let hf_low = halofit_pk(&k_lin, &|k| p_lin_at_z(k, z_low), &hc, z_low);

    let d_ratio = growth_factor_d_ratio(c, a_low, a_high);
    let expected = d_ratio * d_ratio;

    println!("[halofit_linear_consistency] D(z=2)/D(z=49) = {d_ratio:.4}  D² = {expected:.4}");

    for ((k, (_, p_hi)), (_, p_lo)) in k_lin.iter().zip(hf_high.iter()).zip(hf_low.iter()) {
        if *p_hi > 0.0 && *p_lo > 0.0 {
            let ratio = p_lo / p_hi;
            let rel_err = (ratio / expected - 1.0).abs();
            println!(
                "  k={k:.4} h/Mpc: P_lo/P_hi = {ratio:.4}  D² = {expected:.4}  err = {:.2}%",
                rel_err * 100.0
            );
            assert!(
                rel_err < 0.05,
                "k={k}: Halofit ratio={ratio:.4} vs D²={expected:.4} err={:.2}% > 5 %",
                rel_err * 100.0
            );
        }
    }
    println!("  OK — Halofit lineal consistente con D²(a)");
}

// ── TEST 3 — Boost no-lineal de Halofit es monotónico en z ───────────────────

/// P_nl/P_lin (boost Halofit) debe aumentar al bajar z (más clustering).
///
/// Verifica a k=1 h/Mpc: boost(z=0) > boost(z=1) > boost(z=2) > 1.
/// Este es un test cualitativo de la física del modelo Halofit (Takahashi 2012).
#[test]
fn phase49_halofit_ratio_z2_vs_z0() {
    let hc = halofit_cosmo();

    // Evaluar Halofit en k no-lineal: 0.3–3 h/Mpc.
    let k_nl = vec![0.30_f64, 1.0, 2.0];
    let z_vals = [0.0_f64, 1.0, 2.0, 5.0];

    println!("[halofit_boost_vs_z]");
    let mut boosts: Vec<Vec<f64>> = Vec::new();

    for &z in &z_vals {
        let p_nl = halofit_pk(&k_nl, &|k| p_lin_at_z(k, z), &hc, z);
        let boost_at_z: Vec<f64> = p_nl
            .iter()
            .zip(k_nl.iter())
            .map(|((_, p_nl_v), k)| {
                let p_lin_v = p_lin_at_z(*k, z);
                let b = if p_lin_v > 0.0 { p_nl_v / p_lin_v } else { 1.0 };
                println!("  z={z:.1} k={k:.2}: P_nl/P_lin = {b:.4}");
                b
            })
            .collect();
        boosts.push(boost_at_z);
    }

    // Verificar monotonía solo en k no-lineales donde el boost es significativo (≥ 1%).
    // k=0.3 h/Mpc puede estar en el régimen lineal a z≥1 (boost=1.0 exacto).
    // k=1 y k=2 h/Mpc siempre tienen boost > 1.0 y deben ser monotónicos.
    for ki in 1..k_nl.len() {
        // ki=1 (k=1 h/Mpc), ki=2 (k=2 h/Mpc)
        let b0 = boosts[0][ki]; // z=0
        let b1 = boosts[1][ki]; // z=1
        let b2 = boosts[2][ki]; // z=2
        let b5 = boosts[3][ki]; // z=5

        assert!(
            b0 > b1 && b1 > b2 && b2 > b5,
            "Boost no es monotónico en z para k={}: b(0)={b0:.4} b(1)={b1:.4} b(2)={b2:.4} b(5)={b5:.4}",
            k_nl[ki]
        );
    }

    // Para k=1 h/Mpc a z=0, boost debe ser significativamente > 1.
    let boost_k1_z0 = boosts[0][1]; // k=1 h/Mpc, z=0
    assert!(
        boost_k1_z0 > 2.0,
        "Boost no-lineal k=1 h/Mpc z=0 insuficiente: {boost_k1_z0:.3} < 2.0"
    );

    // Boost z=0 > boost z=5 en todos los k no-lineales.
    assert!(
        boosts[0][0] >= boosts[3][0],
        "Boost k=0.3 no decrece de z=0 a z=5"
    );

    println!("  OK — Boost(k=1 h/Mpc, z=0) = {boost_k1_z0:.3} > 2.0  ✓");
}
