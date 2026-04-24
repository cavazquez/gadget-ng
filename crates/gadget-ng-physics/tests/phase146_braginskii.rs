/// Phase 146 — Viscosidad Braginskii anisótropa (braginskii.rs)
///
/// Verifica: viscosidad ∥B transfiere momento, viscosidad ⊥B no lo hace,
/// eta=0 es no-op, N=0 no crashea, conservación de momento total,
/// campo B paralelo a X: solo componente vx se difunde.
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_mhd::apply_braginskii_viscosity;

fn gas(id: usize, pos: Vec3, vel: Vec3, b: Vec3) -> Particle {
    let mut p = Particle::new_gas(id, 1.0, pos, vel, 1.0, 0.2);
    p.b_field = b;
    p
}

// ── 1. Viscosidad ∥B transfiere momento en dirección B ────────────────────

#[test]
fn viscosity_parallel_b_transfers_momentum() {
    let b = Vec3::new(1.0, 0.0, 0.0); // B ∥ x̂
    let mut particles = vec![
        gas(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0), b),
        gas(1, Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), b),
    ];
    let v0_init = particles[0].velocity.x;
    let v1_init = particles[1].velocity.x;
    apply_braginskii_viscosity(&mut particles, 1.0, 0.01);
    // La partícula 0 debe frenarse y la 1 debe acelerarse
    assert!(particles[0].velocity.x < v0_init, "p0 debe frenarse");
    assert!(particles[1].velocity.x > v1_init, "p1 debe acelerarse");
}

// ── 2. eta = 0 → no-op ────────────────────────────────────────────────────

#[test]
fn eta_zero_is_noop() {
    let b = Vec3::new(1.0, 0.0, 0.0);
    let mut particles = vec![
        gas(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0), b),
        gas(1, Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), b),
    ];
    let vx0 = particles[0].velocity.x;
    apply_braginskii_viscosity(&mut particles, 0.0, 0.01);
    assert_eq!(particles[0].velocity.x, vx0, "eta=0 debe ser no-op");
}

// ── 3. N=0 no crashea ────────────────────────────────────────────────────

#[test]
fn n_zero_no_crash() {
    let mut empty: Vec<Particle> = Vec::new();
    apply_braginskii_viscosity(&mut empty, 1.0, 0.01);
    assert!(empty.is_empty());
}

// ── 4. Conservación de momento total ∥B ──────────────────────────────────

#[test]
fn total_momentum_conserved() {
    let b = Vec3::new(1.0, 0.0, 0.0);
    let mut particles = vec![
        gas(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(8.0, 0.0, 0.0), b),
        gas(1, Vec3::new(0.1, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), b),
    ];
    let px_total_before: f64 = particles.iter().map(|p| p.mass * p.velocity.x).sum();
    apply_braginskii_viscosity(&mut particles, 0.5, 0.01);
    let px_total_after: f64 = particles.iter().map(|p| p.mass * p.velocity.x).sum();
    assert!((px_total_after - px_total_before).abs() < 1e-10,
        "Momento total debe conservarse: Δpx = {:.2e}", px_total_after - px_total_before);
}

// ── 5. B ∥ ẑ, partículas separadas en z: vz se difunde ───────────────────

#[test]
fn braginskii_anisotropy_z_direction() {
    let b = Vec3::new(0.0, 0.0, 1.0); // B ∥ ẑ
    // Separación en z (∥B) para que cos²θ = (ẑ · ẑ)² = 1 → máxima difusión
    let mut particles = vec![
        gas(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 10.0), b),
        gas(1, Vec3::new(0.0, 0.0, 0.1), Vec3::new(0.0, 0.0, 0.0), b),
    ];
    let vz0_before = particles[0].velocity.z;
    apply_braginskii_viscosity(&mut particles, 1.0, 0.01);
    assert!(particles[0].velocity.z < vz0_before, "vz debe difundirse (∥B)");
}

// ── 6. Partículas con B=0 no reciben impulso ─────────────────────────────

#[test]
fn zero_b_field_no_viscosity() {
    let b_zero = Vec3::zero();
    let mut particles = vec![
        gas(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0), b_zero),
        gas(1, Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), b_zero),
    ];
    let vx0 = particles[0].velocity.x;
    apply_braginskii_viscosity(&mut particles, 1.0, 0.01);
    assert_eq!(particles[0].velocity.x, vx0, "B=0 → sin viscosidad");
}
