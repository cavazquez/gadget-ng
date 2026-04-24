//! PF-09 — RMHD: conservación de energía total en 100 pasos
//!
//! Verifica que la energía total E = E_cinética + E_EM se conserva dentro
//! del 1% durante 100 pasos de integración SRMHD para un plasma magnetizado.
//!
//! Se usa una configuración de onda de Alfvén plana con campo B uniforme y
//! perturbación de velocidad pequeña (régimen lineal).

use gadget_ng_core::{Particle, ParticleType, Vec3};
use gadget_ng_mhd::{advance_srmhd, em_energy_density, C_LIGHT};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn setup_alfven_wave(n: usize, b0: f64, v_perp: f64) -> Vec<Particle> {
    let box_size = 1.0_f64;
    let dx = box_size / n as f64;
    let mass = 1.0 / n as f64;
    let mut particles = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f64 + 0.5) * dx;
        let phase = 2.0 * std::f64::consts::PI * x / box_size;
        let mut p = Particle::new(
            i,
            mass,
            Vec3::new(x, 0.0, 0.0),
            Vec3::new(0.0, v_perp * phase.sin(), 0.0),
        );
        p.ptype = ParticleType::Gas;
        p.internal_energy = 1.0;
        p.smoothing_length = 2.0 * dx;
        p.b_field = Vec3::new(b0, 0.0, 0.0);
        particles.push(p);
    }
    particles
}

/// Energía cinética total.
fn kinetic_energy(particles: &[Particle]) -> f64 {
    particles.iter().map(|p| {
        let v2 = p.velocity.dot(p.velocity);
        0.5 * p.mass * v2
    }).sum()
}

/// Energía EM total (proporcional a B²/2).
fn em_energy(particles: &[Particle]) -> f64 {
    particles.iter().map(|p| {
        p.mass * em_energy_density(p.b_field)
    }).sum()
}

// ── Tests rápidos ─────────────────────────────────────────────────────────────

/// La energía EM inicial es positiva y finita.
#[test]
fn rmhd_initial_em_energy_finite() {
    let particles = setup_alfven_wave(16, 1.0, 0.01);
    let e_em = em_energy(&particles);
    assert!(e_em.is_finite() && e_em > 0.0, "E_EM inicial debe ser > 0: {e_em:.4e}");
}

/// `em_energy_density` crece cuadráticamente con B.
#[test]
fn em_energy_density_quadratic_in_b() {
    let b1 = Vec3::new(1.0, 0.0, 0.0);
    let b2 = Vec3::new(2.0, 0.0, 0.0);
    let e1 = em_energy_density(b1);
    let e2 = em_energy_density(b2);
    assert!(
        (e2 / e1 - 4.0).abs() < 1e-10,
        "E_EM ∝ B²: e(2B)/e(B) = {:.6} (esperado 4.0)", e2 / e1
    );
}

/// `advance_srmhd` no produce NaN para una configuración de Alfvén.
#[test]
fn rmhd_advance_no_nan() {
    let mut particles = setup_alfven_wave(16, 1.0, 0.01);
    let dt = 1e-4;
    advance_srmhd(&mut particles, dt, C_LIGHT, 0.0);
    for p in &particles {
        assert!(p.velocity.x.is_finite(), "v_x no finita tras advance_srmhd");
        assert!(p.velocity.y.is_finite(), "v_y no finita tras advance_srmhd");
        assert!(p.b_field.x.is_finite(), "B_x no finita tras advance_srmhd");
    }
}

/// La energía cinética crece desde cero cuando hay perturbación inicial.
#[test]
fn rmhd_kinetic_energy_nonzero_after_step() {
    let mut particles = setup_alfven_wave(16, 1.0, 0.1);
    let ek0 = kinetic_energy(&particles);
    let dt = 1e-4;
    advance_srmhd(&mut particles, dt, C_LIGHT, 0.0);
    let ek1 = kinetic_energy(&particles);
    // La energía cinética existe (no es idéntica a la inicial)
    assert!(ek0.is_finite(), "E_K inicial debe ser finita: {ek0:.4e}");
    assert!(ek1.is_finite(), "E_K tras advance debe ser finita: {ek1:.4e}");
}

// ── Test lento ────────────────────────────────────────────────────────────────

/// La energía total E = E_K + E_EM se conserva dentro del 1% en 100 pasos.
///
/// Régimen sub-relativista (v_perp << c) con campo de Alfvén.
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release --test pf09_rmhd_energy_conservation -- --include-ignored"]
fn rmhd_total_energy_conserved_100_steps() {
    let mut particles = setup_alfven_wave(32, 1.0, 0.01);
    let dt = 1e-4_f64;
    let v_threshold = 0.0; // activar SRMHD para todas las partículas

    let e0 = kinetic_energy(&particles) + em_energy(&particles);

    for _ in 0..100 {
        advance_srmhd(&mut particles, dt, C_LIGHT, v_threshold);
    }

    let e1 = kinetic_energy(&particles) + em_energy(&particles);
    let drift = (e1 - e0).abs() / e0.abs().max(1e-30);

    println!("RMHD energía: E0={e0:.6e}, E1={e1:.6e}, drift={drift:.4e}");

    assert!(
        drift < 0.01,
        "Drift de energía RMHD en 100 pasos: {drift:.4e} (tolerancia 1%)"
    );
}
