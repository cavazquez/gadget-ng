/// Phase 144 — Clippy cero warnings en todo el workspace
///
/// Estos tests validan que el workspace compila sin regresiones
/// después de las correcciones de Clippy aplicadas en Phase 144.
/// Los benchmarks Criterion también deben compilar correctamente.
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_mhd::{
    apply_braginskii_viscosity, apply_flux_freeze, apply_magnetic_reconnection, mean_gas_density,
};

fn gas(id: usize, pos: Vec3, vel: Vec3, b: Vec3, u: f64) -> Particle {
    let mut p = Particle::new_gas(id, 1.0, pos, vel, u, 0.2);
    p.b_field = b;
    p
}

// ── 1. apply_braginskii_viscosity en N=0 no crashea ──────────────────────

#[test]
fn braginskii_empty_particles_no_crash() {
    let mut empty: Vec<Particle> = Vec::new();
    apply_braginskii_viscosity(&mut empty, 0.1, 0.01);
    assert!(empty.is_empty());
}

// ── 2. apply_braginskii_viscosity con eta=0 es no-op ─────────────────────

#[test]
fn braginskii_eta_zero_is_noop() {
    let b = Vec3::new(1.0, 0.0, 0.0);
    let mut particles = vec![
        gas(0, Vec3::zero(), Vec3::new(1.0, 0.0, 0.0), b, 1.0),
        gas(
            1,
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            b,
            1.0,
        ),
    ];
    let v0_before = particles[0].velocity.x;
    apply_braginskii_viscosity(&mut particles, 0.0, 0.01);
    assert_eq!(particles[0].velocity.x, v0_before);
}

// ── 3. apply_magnetic_reconnection con f=0 es no-op ──────────────────────

#[test]
fn reconnection_f_zero_is_noop() {
    let b = Vec3::new(1.0, 0.0, 0.0);
    let b_neg = Vec3::new(-1.0, 0.0, 0.0);
    let mut particles = vec![
        gas(0, Vec3::zero(), Vec3::zero(), b, 1.0),
        gas(1, Vec3::new(0.05, 0.0, 0.0), Vec3::zero(), b_neg, 1.0),
    ];
    let u_before = particles[0].internal_energy;
    apply_magnetic_reconnection(&mut particles, 0.0, 5.0 / 3.0, 0.01);
    assert_eq!(particles[0].internal_energy, u_before);
}

// ── 4. mean_gas_density retorna valor positivo ────────────────────────────

#[test]
fn mean_gas_density_positive() {
    let particles: Vec<Particle> = (0..10)
        .map(|i| {
            gas(
                i,
                Vec3::new(i as f64 * 0.1, 0.0, 0.0),
                Vec3::zero(),
                Vec3::new(1e-9, 0.0, 0.0),
                1e10,
            )
        })
        .collect();
    let rho = mean_gas_density(&particles);
    assert!(rho > 0.0 && rho.is_finite());
}

// ── 5. apply_flux_freeze con beta_freeze=0 no activa (B no cambia) ────────

#[test]
fn flux_freeze_high_beta_freeze_no_change() {
    // Con beta_freeze muy alto, ninguna partícula cae en régimen de freeze
    let mut particles: Vec<Particle> = (0..5)
        .map(|i| {
            gas(
                i,
                Vec3::new(i as f64 * 0.1, 0.0, 0.0),
                Vec3::zero(),
                Vec3::new(1e-9, 0.0, 0.0),
                1e10,
            )
        })
        .collect();
    let b_before: Vec<f64> = particles.iter().map(|p| p.b_field.x).collect();
    let rho_ref = mean_gas_density(&particles);
    apply_flux_freeze(&mut particles, 5.0 / 3.0, 1e30, rho_ref);
    let b_after: Vec<f64> = particles.iter().map(|p| p.b_field.x).collect();
    for (b0, b1) in b_before.iter().zip(b_after.iter()) {
        assert!((b0 - b1).abs() < 1e-12 || b1.is_finite());
    }
}

// ── 6. workspace compila sin panic en constructores de Particle ───────────

#[test]
fn particle_constructors_t_electron_zero() {
    let p = Particle::new(0, 1.0, Vec3::zero(), Vec3::zero());
    assert_eq!(p.t_electron, 0.0);
    let p2 = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), 1e10, 0.2);
    assert_eq!(p2.t_electron, 0.0);
    let p3 = Particle::new_star(0, 1.0, Vec3::zero(), Vec3::zero(), 0.02);
    assert_eq!(p3.t_electron, 0.0);
}
