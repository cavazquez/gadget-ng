//! PF-10 — Plasma de dos fluidos: equilibrio termal T_e → T_i
//!
//! Verifica que tras muchos tiempos de equilibración Coulomb, los electrones
//! y los iones alcanzan la misma temperatura:
//!
//! ```text
//! |T_e/T_i - 1| < 0.1%   después de 10 × t_eq
//! ```
//!
//! donde `t_eq = 1 / ν_ei`.

use gadget_ng_core::{Particle, ParticleType, TwoFluidSection, Vec3};
use gadget_ng_mhd::{apply_electron_ion_coupling, mean_te_over_ti};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn gas_particle(id: usize, u_ion: f64, t_electron: f64) -> Particle {
    let mut p = Particle::new(id, 1.0, Vec3::zero(), Vec3::zero());
    p.ptype = ParticleType::Gas;
    p.internal_energy = u_ion;
    p.smoothing_length = 0.1;
    p.t_electron = t_electron;
    p
}

fn active_cfg(nu_ei_coeff: f64) -> TwoFluidSection {
    TwoFluidSection {
        enabled: true,
        nu_ei_coeff,
        t_e_init_k: 0.0,
    }
}

/// T_i a partir de energía interna (γ = 5/3).
fn t_ion(u: f64) -> f64 {
    (2.0 / 3.0) * u
}

// ── Tests rápidos ─────────────────────────────────────────────────────────────

/// El acoplamiento Coulomb reduce la brecha T_e/T_i.
#[test]
fn two_fluid_coupling_reduces_gap() {
    let t_e_init = 2.0_f64;
    let u_i_init = 1.0_f64;
    let mut particles = vec![gas_particle(0, u_i_init, t_e_init)];
    let cfg = active_cfg(10.0);
    let dt = 0.01_f64;

    let gap_0 = (t_e_init - t_ion(u_i_init)).abs();
    apply_electron_ion_coupling(&mut particles, &cfg, dt);
    let gap_1 = (particles[0].t_electron - t_ion(particles[0].internal_energy)).abs();

    assert!(
        gap_1 < gap_0,
        "La brecha T_e-T_i debe reducirse: Δ_0={gap_0:.4}, Δ_1={gap_1:.4}"
    );
}

/// `mean_te_over_ti` devuelve 1.0 cuando T_e = T_i.
#[test]
fn mean_te_over_ti_unity_at_equilibrium() {
    let u = 1.5_f64;
    let t_i = t_ion(u);
    let particles = vec![
        gas_particle(0, u, t_i),
        gas_particle(1, u, t_i),
    ];
    let ratio = mean_te_over_ti(&particles);
    assert!(
        (ratio - 1.0).abs() < 1e-10,
        "T_e/T_i debe ser 1.0 en equilibrio: {ratio:.6}"
    );
}

/// El acoplamiento es noop para partículas de materia oscura.
#[test]
fn two_fluid_coupling_ignores_dm() {
    let mut p = Particle::new(0, 1.0, Vec3::zero(), Vec3::zero());
    p.ptype = ParticleType::DarkMatter;
    p.t_electron = 5.0;
    p.internal_energy = 1.0;
    let te0 = p.t_electron;
    let u0 = p.internal_energy;
    let cfg = active_cfg(100.0);
    let mut particles = vec![p];
    apply_electron_ion_coupling(&mut particles, &cfg, 1.0);
    assert_eq!(particles[0].t_electron, te0, "DM no debe verse afectada");
    assert_eq!(particles[0].internal_energy, u0, "DM no debe verse afectada");
}

/// T_e permanece positiva después del acoplamiento.
#[test]
fn two_fluid_te_stays_positive() {
    let mut particles = vec![gas_particle(0, 1.0, 0.001)]; // T_e << T_i
    let cfg = active_cfg(10.0);
    for _ in 0..100 {
        apply_electron_ion_coupling(&mut particles, &cfg, 0.01);
    }
    assert!(
        particles[0].t_electron > 0.0,
        "T_e debe permanecer positiva: {}", particles[0].t_electron
    );
}

// ── Test lento ────────────────────────────────────────────────────────────────

/// |T_e/T_i - 1| < 0.1% después de 10 × t_eq (Coulomb).
///
/// Se usa ν_ei = 10, de modo que t_eq ≈ 0.1. En dt = 0.001, 10 t_eq ≈ 1000 pasos.
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release --test pf10_two_fluid_equilibrium -- --include-ignored"]
fn two_fluid_reaches_thermal_equilibrium_long_time() {
    let nu_ei = 10.0_f64;
    let t_eq = 1.0 / nu_ei;
    let n_teq = 10.0_f64; // tiempos de equilibración
    let dt = 0.001_f64;
    let n_steps = (n_teq * t_eq / dt).ceil() as usize;

    let t_e_init = 2.0_f64;
    let u_i_init = 1.0_f64; // T_i ≈ 2/3 → lejos del equilibrio

    // Usar N=16 partículas para estadística
    let mut particles: Vec<Particle> = (0..16)
        .map(|i| gas_particle(i, u_i_init, t_e_init))
        .collect();

    let cfg = active_cfg(nu_ei);

    for _ in 0..n_steps {
        apply_electron_ion_coupling(&mut particles, &cfg, dt);
    }

    let ratio = mean_te_over_ti(&particles);
    println!(
        "Two-fluid equilibrio: T_e/T_i = {ratio:.6} tras {n_steps} pasos ({n_teq} t_eq)"
    );

    assert!(
        (ratio - 1.0).abs() < 0.001,
        "|T_e/T_i - 1| = {:.4e} (tolerancia 0.1%)", (ratio - 1.0).abs()
    );
}
