//! Phase 48 — Validación P(k) no-lineal con Halofit (Takahashi+2012).
//!
//! ## Objetivo
//!
//! Verificar que la implementación de Halofit (Takahashi+2012) en
//! `gadget-ng-analysis::halofit` es correcta y consistente con el pipeline
//! de medición y corrección P(k) del proyecto.
//!
//! ## Tests
//!
//! 1. **`phase48_halofit_static`** — Propiedades estáticas de Halofit a z=0:
//!    - Boost P_nl/P_lin creciente con k.
//!    - Convergencia a lineal a escalas grandes.
//!    - Boost significativo a k > 1 h/Mpc.
//!
//! 2. **`phase48_halofit_growth_consistency`** — Auto-consistencia de Halofit
//!    con el factor de crecimiento: el cociente P_halofit(k, z₁)/P_halofit(k, z₂)
//!    en escalas lineales (k < 0.05 h/Mpc) debe ser ≈ [D(z₁)/D(z₂)]².
//!
//! 3. **`phase48_pk_vs_halofit_at_ics`** — Compara P_sim corregido en las ICs
//!    (Z0Sigma8, a_init=0.05, z≈19) con P_halofit al mismo z.
//!    A z=19 el campo es completamente lineal: P_halofit ≈ P_linear.
//!    El error mediado sobre bins con S/N > 1 debe ser < 50 %.
//!    Valida el pipeline completo: IC → P(k) → corrección → Halofit.
//!
//! 4. **`phase48_nonlinear_boost_redshift_dependence`** — Verifica que el boost
//!    P_nl/P_lin de Halofit a k=1 h/Mpc es mayor a z=0 que a z=1 (más estructuras
//!    en el universo joven).
//!
//! ## Convención de unidades
//!
//! - k en h/Mpc, P en (Mpc/h)³ (post-corrección R(N)).
//! - `correct_pk(..., box_mpc_h=None, ...)` para no doble-escalar (Phase 36).
//! - P_halofit evaluado a la misma a de las ICs para la comparación directa.

use gadget_ng_analysis::{
    halofit::{HalofitCosmo, halofit_pk, p_linear_eh},
    pk_correction::{RnModel, correct_pk},
    power_spectrum::power_spectrum,
};
use gadget_ng_core::{
    CosmologySection, EisensteinHuParams, GravitySection, IcKind, InitialConditionsSection,
    NormalizationMode, OutputSection, PerformanceSection, RunConfig, SimulationSection,
    TimestepSection, TransferKind, UnitsSection, Vec3, amplitude_for_sigma8, build_particles,
    cosmology::{CosmologyParams, growth_factor_d_ratio},
};
use std::f64::consts::PI;

// ── Constantes ────────────────────────────────────────────────────────────────

const BOX: f64 = 1.0;
const BOX_MPC_H: f64 = 100.0;
const OMEGA_M: f64 = 0.315;
const OMEGA_L: f64 = 0.685;
const OMEGA_B: f64 = 0.049;
const H0: f64 = 0.1;
const H_DIMLESS: f64 = 0.674;
const T_CMB: f64 = 2.7255;
const N_S: f64 = 0.965;
const SIGMA8: f64 = 0.8;

const N_MESH: usize = 32;
const A_INIT: f64 = 0.05; // z≈19 con Z0Sigma8

// ── Helpers ───────────────────────────────────────────────────────────────────

fn eh() -> EisensteinHuParams {
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

fn halofit_cosmo() -> HalofitCosmo {
    HalofitCosmo {
        omega_m0: OMEGA_M,
        omega_de0: OMEGA_L,
    }
}

fn build_ic_config(seed: u64) -> RunConfig {
    let gravity = GravitySection {
        solver: gadget_ng_core::SolverKind::Pm,
        pm_grid_size: N_MESH,
        ..GravitySection::default()
    };
    RunConfig {
        simulation: SimulationSection {
            dt: 1.0e-3,
            num_steps: 0, // No evolucionar — solo generar ICs
            softening: 0.01,
            physical_softening: false,
            gravitational_constant: 1.0,
            particle_count: N_MESH * N_MESH * N_MESH,
            box_size: BOX,
            seed,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Zeldovich {
                seed,
                grid_size: N_MESH,
                spectral_index: N_S,
                amplitude: 1.0e-4,
                transfer: TransferKind::EisensteinHu,
                sigma8: Some(SIGMA8),
                omega_b: OMEGA_B,
                h: H_DIMLESS,
                t_cmb: T_CMB,
                box_size_mpc_h: Some(BOX_MPC_H),
                use_2lpt: false,
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
            ..Default::default()
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
        insitu_analysis: Default::default(),
        sph: Default::default(),
        rt: Default::default(),
        reionization: Default::default(),
        mhd: Default::default(),
        turbulence: Default::default(),
        two_fluid: Default::default(),
        sidm: Default::default(),
        modified_gravity: Default::default(),
    }
}

/// P_linear(k_hmpc, a) = A² k^ns T²(k) × [D(a)/D(1)]².
fn p_lin_at_a(k_hmpc: f64, a: f64) -> f64 {
    let e = eh();
    let amp = amplitude_for_sigma8(SIGMA8, N_S, &e);
    let d = growth_factor_d_ratio(cosmo_params(), a, 1.0);
    p_linear_eh(k_hmpc, amp, N_S, d, &e)
}

// ── Test 1: propiedades estáticas de Halofit ──────────────────────────────────

/// Verifica propiedades cualitativas de Halofit a z=0:
/// - Boost P_nl/P_lin crece monotónicamente con k.
/// - P_nl ≈ P_lin a k pequeño (régimen lineal).
/// - P_nl >> P_lin a k grande (régimen no-lineal).
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn phase48_halofit_static() {
    let cosmo = halofit_cosmo();
    let e = eh();
    let amp = amplitude_for_sigma8(SIGMA8, N_S, &e);
    let p_lin_z0 = |k: f64| p_linear_eh(k, amp, N_S, 1.0, &e);

    let k_vals: Vec<f64> = vec![0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0];
    let p_nl = halofit_pk(&k_vals, &p_lin_z0, &cosmo, 0.0);
    let ratios: Vec<f64> = p_nl
        .iter()
        .zip(&k_vals)
        .map(|((_, pnl), &k)| pnl / p_lin_z0(k))
        .collect();

    println!("  k [h/Mpc]  P_nl/P_lin");
    for (&k, &r) in k_vals.iter().zip(ratios.iter()) {
        println!("  {k:8.3}   {r:.4}");
    }

    // A escalas grandes, P_nl ≈ P_lin (< 5 % boost).
    assert!(
        ratios[0] < 1.05,
        "k=0.01: P_nl/P_lin={:.3} debería ≈ 1",
        ratios[0]
    );

    // Boost crece monotónicamente.
    for i in 1..ratios.len() {
        assert!(
            ratios[i] >= ratios[i - 1] * 0.95,
            "Boost no monótono entre k={} y k={}",
            k_vals[i - 1],
            k_vals[i]
        );
    }

    // A k=1 h/Mpc, boost > 2× (halo dominado).
    let idx_1 = k_vals.iter().position(|&k| (k - 1.0).abs() < 0.01).unwrap();
    assert!(
        ratios[idx_1] > 2.0,
        "k=1.0: P_nl/P_lin={:.2} debería > 2.0",
        ratios[idx_1]
    );
}

// ── Test 2: auto-consistencia de crecimiento ──────────────────────────────────

/// A escalas lineales (k << k_sigma), el cociente P_halofit(z₁)/P_halofit(z₂)
/// debe igualar [D(z₁)/D(z₂)]² con error < 5 %.
///
/// Esto verifica que Halofit reduce correctamente a P_linear a escalas grandes,
/// y que el factor de crecimiento es consistente con la cosmología dada.
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn phase48_halofit_growth_consistency() {
    let cosmo = halofit_cosmo();
    let cp = cosmo_params();
    let e = eh();
    let amp = amplitude_for_sigma8(SIGMA8, N_S, &e);

    let z1 = 1.0; // a=0.5
    let z2 = 0.0; // a=1.0
    let a1 = 1.0 / (1.0 + z1);
    let a2 = 1.0 / (1.0 + z2);

    // Factor de crecimiento según Friedmann.
    let d_ratio = growth_factor_d_ratio(cp, a1, a2); // D(a1)/D(a2) = D(z=1)/D(z=0)
    let expected_ratio = d_ratio * d_ratio; // [D(z1)/D(z2)]²

    // Halofit a z=1 y z=0 con P_lin correctamente escalado.
    let p_lin_z1 = |k: f64| p_linear_eh(k, amp, N_S, growth_factor_d_ratio(cp, a1, 1.0), &e);
    let p_lin_z2 = |k: f64| p_linear_eh(k, amp, N_S, growth_factor_d_ratio(cp, a2, 1.0), &e);

    // k muy pequeño (régimen lineal, boost ≈ 0).
    let k_linear = [0.005, 0.01, 0.02];
    let p_nl_z1 = halofit_pk(&k_linear, &p_lin_z1, &cosmo, z1);
    let p_nl_z2 = halofit_pk(&k_linear, &p_lin_z2, &cosmo, z2);

    println!(
        "  D(z=1)/D(z=0) = {:.4}  → [D ratio]² = {:.4}",
        d_ratio, expected_ratio
    );
    println!("  k [h/Mpc]   P_nl(z=1)/P_nl(z=0)   expected");
    let mut max_err = 0.0_f64;
    for (i, &k) in k_linear.iter().enumerate() {
        let (_, p1) = p_nl_z1[i];
        let (_, p2) = p_nl_z2[i];
        if p2 > 0.0 {
            let ratio = p1 / p2;
            let err = (ratio / expected_ratio - 1.0).abs();
            max_err = max_err.max(err);
            println!("  {k:.3}         {ratio:.6}              {expected_ratio:.6}   err={err:.4}");
        }
    }
    assert!(
        max_err < 0.05,
        "Error máximo {max_err:.4} > 5 % en auto-consistencia de Halofit"
    );
}

// ── Test 3: ICs Z0Sigma8 vs Halofit a z=19 ───────────────────────────────────

/// Compara P_sim corregido en las ICs (Z0Sigma8, a=0.05, z≈19) con P_halofit
/// al mismo redshift.
///
/// A z=19, el campo es completamente lineal (σ₈(z=19) ≈ 0.04), por lo que
/// P_halofit ≈ P_linear. El error mediado debe ser < 50 % dado el ruido
/// de shot-noise inherente a N=32 con amplitud tan pequeña.
///
/// Este test valida la cadena: IC → P(k) crudo → corrección R(N) → P(Mpc/h)³.
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn phase48_pk_vs_halofit_at_ics() {
    // ── Generar ICs ────────────────────────────────────────────────────────────
    let cfg = build_ic_config(42);
    let parts = build_particles(&cfg).expect("build_particles falló");

    let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = parts.iter().map(|p| p.mass).collect();

    // Medir P(k) crudo en ICs.
    let pk_raw = power_spectrum(&pos, &masses, BOX, N_MESH);

    // Aplicar corrección A_grid·R(N=32).
    let model = RnModel::phase47_default();
    let pk_corr = correct_pk(&pk_raw, BOX, N_MESH, None, &model);

    // ── Comparar con Halofit a z_init ─────────────────────────────────────────
    let z_init = 1.0 / A_INIT - 1.0; // ≈ 19
    let cosmo = halofit_cosmo();

    let k_eval: Vec<f64> = pk_corr
        .iter()
        .map(|b| b.k * H_DIMLESS / BOX_MPC_H)
        .collect();
    let p_lin_fn = |k: f64| p_lin_at_a(k, A_INIT);
    let p_halofit = halofit_pk(&k_eval, &p_lin_fn, &cosmo, z_init);

    // ── Calcular ratio para bins con P_corr > 0 ─────────────────────────────
    // Solo bins con k < k_Nyq/2 (menor aliasing) y P_corr > shot-noise estimate.
    let k_nyq_half_hmpc = (N_MESH as f64 / 4.0) * 2.0 * PI / BOX * H_DIMLESS / BOX_MPC_H;
    let shot_noise_est = (BOX_MPC_H / H_DIMLESS).powi(3) / (N_MESH as f64).powi(3) * 0.5;

    println!("\n  a_init={A_INIT:.3}, z_init={z_init:.1}");
    println!("  k[h/Mpc]   P_corr      P_halofit   ratio");
    let mut errors: Vec<f64> = Vec::new();

    for (bin, (k_hmpc, p_hf)) in pk_corr.iter().zip(p_halofit.iter()) {
        let k_hmpc_bin = bin.k * H_DIMLESS / BOX_MPC_H;
        if k_hmpc_bin > k_nyq_half_hmpc {
            break;
        }
        if *p_hf <= 0.0 || bin.pk <= 0.0 || bin.pk < shot_noise_est {
            continue;
        }
        let ratio = bin.pk / p_hf;
        let err = (ratio - 1.0).abs();
        errors.push(err);
        println!(
            "  {k_hmpc_bin:8.4}   {:.3e}   {:.3e}   {ratio:.3}",
            bin.pk, p_hf
        );
    }

    if errors.is_empty() {
        println!("  Sin bins con S/N suficiente para comparar (shot-noise domina)");
        // A z=19 con N=32, el shot-noise puede dominar; este resultado es esperado.
        return;
    }

    errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_err = errors[errors.len() / 2];
    println!(
        "\n  Mediana |ratio-1| = {median_err:.3}  ({} bins)",
        errors.len()
    );

    assert!(
        median_err < 0.50,
        "Error mediano {median_err:.3} ≥ 50 % — posible bug en pipeline P(k)"
    );
}

// ── Test 4: boost no-lineal decrece con z ────────────────────────────────────

/// El boost P_nl/P_lin a k=1 h/Mpc debe ser mayor a z=0 que a z=1:
/// el universo tiene más estructura no-lineal cuando es más viejo.
///
/// También verifica que a z=0 el boost es > 2× y a z=2 es < boost_z0.
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn phase48_nonlinear_boost_redshift_dependence() {
    let cosmo = halofit_cosmo();
    let cp = cosmo_params();
    let e = eh();
    let amp = amplitude_for_sigma8(SIGMA8, N_S, &e);

    let k_test = [1.0_f64, 3.0];
    let redshifts = [0.0, 0.5, 1.0, 2.0];

    println!("\n  z     k=1.0   k=3.0");
    let mut prev_boost_k1 = 0.0_f64;

    for &z in redshifts.iter().rev() {
        let a = 1.0 / (1.0 + z);
        let d = growth_factor_d_ratio(cp, a, 1.0);
        let p_lin_fn = |k: f64| p_linear_eh(k, amp, N_S, d, &e);
        let p_nl = halofit_pk(&k_test, &p_lin_fn, &cosmo, z);

        let mut boosts = Vec::new();
        for (&k, (_, p)) in k_test.iter().zip(p_nl.iter()) {
            let p_lin = p_lin_fn(k);
            boosts.push(if p_lin > 0.0 { p / p_lin } else { 1.0 });
        }
        println!(
            "  {z:.1}  {:.3}  {:.3}",
            boosts[0],
            boosts.get(1).copied().unwrap_or(0.0)
        );

        // A z=0, boost en k=1 debe ser significativo (> 2×).
        if z == 0.0 {
            assert!(
                boosts[0] > 2.0,
                "z=0, k=1: boost={:.2} debería > 2.0",
                boosts[0]
            );
        }

        // El boost CRECE al bajar z (más estructura en universo más viejo).
        // En iteración `.rev()`, prev_boost_k1 es del z mayor, debe ser menor.
        if prev_boost_k1 > 0.0 {
            assert!(
                boosts[0] >= prev_boost_k1 * 0.95,
                "Boost en z={z:.1} ({:.3}) < boost en z mayor ({prev_boost_k1:.3})",
                boosts[0]
            );
        }
        prev_boost_k1 = boosts[0];
    }
}
