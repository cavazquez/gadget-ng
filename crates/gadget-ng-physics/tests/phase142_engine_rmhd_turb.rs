/// Phase 142 — Engine: RMHD + turbulencia integrados en maybe_mhd! y maybe_sph!
///
/// Tests: TwoFluidSection default, turbulencia activa modifica velocidades,
///        flux-freeze en maybe_mhd no crashea, SRMHD con v<threshold no cambia,
///        reconexión en engine modifica B, Braginskii en engine modifica v.
use gadget_ng_core::{MhdSection, TurbulenceSection, TwoFluidSection, Vec3, Particle};
use gadget_ng_mhd::{
    apply_braginskii_viscosity, apply_electron_ion_coupling, apply_flux_freeze,
    apply_magnetic_reconnection, apply_turbulent_forcing, advance_srmhd, C_LIGHT,
};

fn gas(id: usize, pos: Vec3, vel: Vec3, b: Vec3, u: f64) -> Particle {
    let mut p = Particle::new_gas(id, 1.0, pos, vel, u, 0.2);
    p.b_field = b;
    p
}

// ── 1. TwoFluidSection default ────────────────────────────────────────────

#[test]
fn two_fluid_section_defaults() {
    let cfg = TwoFluidSection::default();
    assert!(!cfg.enabled);
    assert!((cfg.nu_ei_coeff - 1.0).abs() < 1e-10);
    assert_eq!(cfg.t_e_init_k, 0.0);
}

// ── 2. apply_turbulent_forcing activa perturba velocidades ────────────────

#[test]
fn turbulent_forcing_modifies_velocities() {
    let cfg = TurbulenceSection {
        enabled: true,
        amplitude: 1e-2,
        correlation_time: 1.0,
        k_min: 1.0,
        k_max: 4.0,
        spectral_index: 5.0 / 3.0,
    };
    let mut particles: Vec<Particle> = (0..8).map(|i| {
        gas(i, Vec3::new(i as f64 * 0.1, 0.0, 0.0), Vec3::zero(), Vec3::new(1.0, 0.0, 0.0), 1e8)
    }).collect();
    apply_turbulent_forcing(&mut particles, &cfg, 0.01, 42);
    let v_sum: f64 = particles.iter().map(|p| p.velocity.x.abs() + p.velocity.y.abs()).sum();
    assert!(v_sum > 0.0, "turbulencia debe perturbar v: {v_sum:.2e}");
}

// ── 3. apply_flux_freeze integrado en engine (no crashea) ─────────────────

#[test]
fn flux_freeze_in_engine_no_crash() {
    let mut particles: Vec<Particle> = (0..5).map(|i| {
        gas(i, Vec3::new(i as f64 * 0.1, 0.0, 0.0),
            Vec3::zero(), Vec3::new(1e-9, 0.0, 0.0), 1e12)
    }).collect();
    let rho_ref = gadget_ng_mhd::mean_gas_density(&particles);
    apply_flux_freeze(&mut particles, 5.0/3.0, 100.0, rho_ref);
    // Debe ejecutar sin panic
    assert!(particles.iter().all(|p| p.b_field.x.is_finite()));
}

// ── 4. advance_srmhd con v < threshold no modifica posición ──────────────

#[test]
fn srmhd_sub_threshold_no_position_change() {
    let v_sub = Vec3::new(0.05 * C_LIGHT, 0.0, 0.0); // v/c = 0.05 < 0.1
    let mut particles = vec![gas(0, Vec3::zero(), v_sub, Vec3::new(1.0, 0.0, 0.0), 1.0)];
    let pos_before = particles[0].position.x;
    advance_srmhd(&mut particles, 0.01, C_LIGHT, 0.1);
    assert_eq!(particles[0].position.x, pos_before, "v < threshold → no corrección relativista");
}

// ── 5. apply_magnetic_reconnection libera calor y reduce B ───────────────

#[test]
fn reconnection_releases_heat_reduces_b() {
    // Partículas adyacentes con B antiparalelo
    let b_pos = Vec3::new(1.0, 0.0, 0.0);
    let b_neg = Vec3::new(-1.0, 0.0, 0.0);
    let mut particles = vec![
        gas(0, Vec3::new(0.0, 0.0, 0.0), Vec3::zero(), b_pos, 1.0),
        gas(1, Vec3::new(0.1, 0.0, 0.0), Vec3::zero(), b_neg, 1.0),
    ];
    let u0_before = particles[0].internal_energy;
    apply_magnetic_reconnection(&mut particles, 0.1, 5.0/3.0, 0.1);
    assert!(particles[0].internal_energy > u0_before, "Reconexión debe calentar gas");
    assert!(particles[0].b_field.x.abs() < b_pos.x.abs(), "Reconexión debe reducir |B|");
}

// ── 6. apply_braginskii_viscosity transfiere momento ∥B ──────────────────

#[test]
fn braginskii_transfers_momentum_parallel_b() {
    let b = Vec3::new(1.0, 0.0, 0.0);
    let mut particles = vec![
        gas(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), b, 1.0),
        gas(1, Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), b, 1.0),
    ];
    let v0_before = particles[0].velocity.x;
    apply_braginskii_viscosity(&mut particles, 0.5, 0.01);
    let v0_after = particles[0].velocity.x;
    assert!(v0_after < v0_before, "Viscosidad Braginskii debe frenar la partícula más rápida");
}
