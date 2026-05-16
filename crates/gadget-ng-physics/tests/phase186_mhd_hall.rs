//! Phase 186 — MHD Hall term.
//!
//! Tests cuantitativos del drift Hall en `gadget_ng_mhd::apply_hall_drift`:
//! 1. Conservación de |B| bajo drift puro.
//! 2. Rotación correcta de la dirección de B para un campo uniforme.
//! 3. No afecta partículas no-gas (DM).
//! 4. Campo intocado cuando η_H = 0.
//! 5. Múltiples pasos conservan |B| acumulativamente.

use gadget_ng_core::{Particle, ParticleType, Vec3};
use gadget_ng_mhd::apply_hall_drift;

fn make_gas(bx: f64, by: f64, bz: f64, vx: f64, vy: f64, vz: f64, h: f64) -> Particle {
    let mut p = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::new(vx, vy, vz), 1.0, h);
    p.b_field = Vec3::new(bx, by, bz);
    p
}

fn make_dm(bx: f64, by: f64, bz: f64) -> Particle {
    let mut p = Particle::new(0, 1.0, Vec3::zero(), Vec3::new(0.0, 1.0, 0.0));
    p.b_field = Vec3::new(bx, by, bz);
    p
}

/// |B| debe conservarse exactamente (rotación de Rodrigues).
#[test]
fn hall_drift_conserves_b_magnitude() {
    let mut particles = vec![
        make_gas(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1),
        make_gas(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.2),
        make_gas(1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.15),
    ];
    let b0: Vec<f64> = particles.iter().map(|p| p.b_field.norm()).collect();

    apply_hall_drift(&mut particles, 0.5, 1.0);

    for (i, p) in particles.iter().enumerate() {
        let b1 = p.b_field.norm();
        assert!(
            (b1 - b0[i]).abs() < 1e-12,
            "partícula {i}: |B| cambió de {:.6e} a {:.6e}",
            b0[i],
            b1
        );
    }
}

/// La dirección de B debe rotar cuando v ⊥ B.
#[test]
fn hall_drift_rotates_b_direction_when_v_perp_b() {
    let mut p = make_gas(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1);
    let b_before = p.b_field;
    apply_hall_drift(std::slice::from_mut(&mut p), 1.0, 1.0);
    let b_after = p.b_field;

    // Componente y o z debe haber aparecido (rotación en el plano x-z o x-y).
    let changed = (b_after.x - b_before.x).abs() > 1e-10
        || (b_after.y - b_before.y).abs() > 1e-10
        || (b_after.z - b_before.z).abs() > 1e-10;
    assert!(changed, "B no rotó a pesar de v ⊥ B");
}

/// B no cambia cuando v ∥ B (v × B = 0 → sin eje de rotación).
#[test]
fn hall_drift_no_rotation_when_v_parallel_b() {
    let mut p = make_gas(1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.1);
    let b_before = p.b_field;
    apply_hall_drift(std::slice::from_mut(&mut p), 1.0, 1.0);
    assert!(
        (p.b_field.x - b_before.x).abs() < 1e-12,
        "B no debería rotar cuando v ∥ B"
    );
}

/// Las partículas DM no deben verse afectadas.
#[test]
fn hall_drift_does_not_affect_dark_matter() {
    let mut particles = vec![
        make_dm(1.0, 0.0, 0.0),
        make_gas(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1),
    ];
    let b_dm_before = particles[0].b_field;
    apply_hall_drift(&mut particles, 0.5, 1.0);
    assert_eq!(particles[0].b_field.x, b_dm_before.x, "DM Bx cambió");
    assert_eq!(particles[0].b_field.y, b_dm_before.y, "DM By cambió");
    assert_eq!(particles[0].b_field.z, b_dm_before.z, "DM Bz cambió");
    assert_eq!(particles[0].ptype, ParticleType::DarkMatter);
}

/// Con η_H = 0 el campo no cambia.
#[test]
fn hall_drift_zero_eta_leaves_b_unchanged() {
    let mut p = make_gas(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1);
    let b_before = p.b_field;
    apply_hall_drift(std::slice::from_mut(&mut p), 0.0, 1.0);
    assert_eq!(p.b_field.x, b_before.x);
    assert_eq!(p.b_field.y, b_before.y);
    assert_eq!(p.b_field.z, b_before.z);
}

/// Múltiples pasos conservan |B| acumulativamente.
#[test]
fn hall_drift_conserves_b_over_many_steps() {
    let mut p = make_gas(1.0, 0.5, 0.2, 0.3, 0.7, -0.1, 0.1);
    let b0 = p.b_field.norm();
    for _ in 0..100 {
        apply_hall_drift(std::slice::from_mut(&mut p), 0.1, 0.01);
    }
    let b1 = p.b_field.norm();
    assert!(
        (b1 - b0).abs() < 1e-10,
        "|B| drift acumulado: {b0:.8} → {b1:.8}"
    );
}

/// La energía interna no cambia (Hall no disipa energía magnética).
#[test]
fn hall_drift_does_not_change_internal_energy() {
    let mut p = make_gas(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1);
    let u0 = p.internal_energy;
    apply_hall_drift(std::slice::from_mut(&mut p), 0.5, 1.0);
    assert_eq!(
        p.internal_energy, u0,
        "energía interna cambió (Hall no debe calentar)"
    );
}
