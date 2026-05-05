//! Fase A roadmap: pérdidas CR y kick polvo radiativo.

use gadget_ng_core::{DustSection, Particle, Vec3};
use gadget_ng_sph::{apply_cr_hadronic_losses, apply_dust_radiation_pressure_kick};

#[test]
fn hadronic_loss_reduces_cr_energy() {
    let mut v = vec![Particle::new_gas(
        0,
        1.0,
        Vec3::zero(),
        Vec3::zero(),
        1.0,
        1.0,
    )];
    v[0].cr_energy = 100.0;
    apply_cr_hadronic_losses(&mut v, 1e-3, 1.0);
    assert!(
        v[0].cr_energy < 100.0,
        "e_cr debe decrecer con pérdidas hadrónicas"
    );
}

#[test]
fn dust_radiation_kick_moves_gas_along_z() {
    let mut v = vec![Particle::new_gas(
        0,
        1.0,
        Vec3::new(1.0, 1.0, 8.0),
        Vec3::zero(),
        1.0,
        0.5,
    )];
    v[0].dust_to_gas = 0.005;
    let cfg = DustSection {
        enabled: true,
        radiation_pressure_enabled: true,
        radiation_pressure_kappa: 1e6,
        radiation_pressure_j_uv: 1e-6,
        ..Default::default()
    };
    let vz0 = v[0].velocity.z;
    apply_dust_radiation_pressure_kick(&mut v, &cfg, 5.0, 0.01);
    assert_ne!(
        v[0].velocity.z, vz0,
        "impulso vertical distinto de cero esperado"
    );
}

#[test]
fn hadronic_zero_is_noop() {
    let mut v = vec![Particle::new_gas(
        0,
        1.0,
        Vec3::zero(),
        Vec3::zero(),
        1.0,
        1.0,
    )];
    v[0].cr_energy = 50.0;
    apply_cr_hadronic_losses(&mut v, 0.0, 1.0);
    assert!((v[0].cr_energy - 50.0).abs() < 1e-15);
}
