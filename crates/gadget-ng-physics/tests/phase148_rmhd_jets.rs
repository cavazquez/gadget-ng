/// Phase 148 — RMHD cosmológica: jets AGN relativistas desde halos FoF
///
/// Tests: jet inyecta v = v_jet, B alineado, energía interna aumenta,
///        jet bipolar (+z/-z), n_jet=0 es no-op, Lorentz factor de jet.
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_mhd::{C_LIGHT, inject_relativistic_jet, lorentz_factor};

fn gas(id: usize, pos: Vec3) -> Particle {
    Particle::new_gas(id, 1e10, pos, Vec3::zero(), 1e6, 0.1)
}

// ── 1. Jet inyecta v = v_jet en partícula más cercana ────────────────────

#[test]
fn jet_injects_velocity() {
    let center = Vec3::new(0.0, 0.0, 0.0);
    let halo_centers = vec![center];
    let mut particles = vec![
        gas(0, Vec3::new(0.0, 0.0, 0.1)),  // z > 0 → jet +z
        gas(1, Vec3::new(0.0, 0.0, -0.1)), // z < 0 → jet -z
    ];
    let v_frac = 0.3;
    inject_relativistic_jet(&mut particles, &halo_centers, v_frac, 1, C_LIGHT, 1.0);
    let v_jet_expected = v_frac * C_LIGHT;
    assert!(
        (particles[0].velocity.z - v_jet_expected).abs() < 1.0,
        "p0 vz = {:.3e} ≈ {v_jet_expected:.3e}",
        particles[0].velocity.z
    );
    assert!(
        (particles[1].velocity.z + v_jet_expected).abs() < 1.0,
        "p1 vz = {:.3e} ≈ {:.3e}",
        particles[1].velocity.z,
        -v_jet_expected
    );
}

// ── 2. Jet alinea B con la dirección del jet ──────────────────────────────

#[test]
fn jet_aligns_b_with_velocity() {
    let halo_centers = vec![Vec3::zero()];
    let mut particles = vec![
        gas(0, Vec3::new(0.0, 0.0, 0.5)),
        gas(1, Vec3::new(0.0, 0.0, -0.5)),
    ];
    inject_relativistic_jet(&mut particles, &halo_centers, 0.5, 1, C_LIGHT, 1e-6);
    // B del jet + debe apuntar en +z, B del jet - en -z
    assert!(
        particles[0].b_field.z > 0.0,
        "B_jet debe apuntar en +z para p0"
    );
    assert!(
        particles[1].b_field.z < 0.0,
        "B_jet debe apuntar en -z para p1"
    );
}

// ── 3. Energía interna del jet es E = (γ−1)mc² ───────────────────────────

#[test]
fn jet_energy_relativistic() {
    let halo_centers = vec![Vec3::zero()];
    let mut particles = vec![
        gas(0, Vec3::new(0.0, 0.0, 1.0)),
        gas(1, Vec3::new(0.0, 0.0, -1.0)),
    ];
    let v_frac = 0.9;
    inject_relativistic_jet(&mut particles, &halo_centers, v_frac, 1, C_LIGHT, 1.0);
    let v_jet = Vec3::new(0.0, 0.0, v_frac * C_LIGHT);
    let gamma = lorentz_factor(v_jet, C_LIGHT);
    let u_expected = (gamma - 1.0) * C_LIGHT * C_LIGHT;
    assert!(
        particles[0].internal_energy >= u_expected * 0.9,
        "u_jet = {:.3e} ≥ 0.9 × {u_expected:.3e}",
        particles[0].internal_energy
    );
}

// ── 4. n_jet_halos = 0 → no-op ────────────────────────────────────────────

#[test]
fn zero_halos_no_jet() {
    let halo_centers = vec![Vec3::zero()];
    let mut particles = vec![gas(0, Vec3::new(0.0, 0.0, 0.1))];
    let v0_before = particles[0].velocity.z;
    inject_relativistic_jet(&mut particles, &halo_centers, 0.3, 0, C_LIGHT, 1.0);
    assert_eq!(particles[0].velocity.z, v0_before, "n_jet=0 debe ser no-op");
}

// ── 5. v_jet = 0 → no-op ──────────────────────────────────────────────────

#[test]
fn zero_v_jet_no_injection() {
    let halo_centers = vec![Vec3::zero()];
    let mut particles = vec![gas(0, Vec3::new(0.0, 0.0, 0.1))];
    let u_before = particles[0].internal_energy;
    inject_relativistic_jet(&mut particles, &halo_centers, 0.0, 1, C_LIGHT, 1.0);
    assert_eq!(
        particles[0].internal_energy, u_before,
        "v_jet=0 debe ser no-op"
    );
}

// ── 6. Factor de Lorentz del jet es consistente ───────────────────────────

#[test]
fn lorentz_factor_jet_consistent() {
    // γ(v=0.9c) ≈ 2.294
    let v = Vec3::new(0.0, 0.0, 0.9 * C_LIGHT);
    let gamma = lorentz_factor(v, C_LIGHT);
    let expected = 1.0 / (1.0 - 0.81_f64).sqrt();
    assert!(
        (gamma - expected).abs() < 0.001,
        "γ(0.9c) = {gamma:.4} ≠ {expected:.4}"
    );
    assert!(gamma > 1.0, "γ siempre > 1");
}
