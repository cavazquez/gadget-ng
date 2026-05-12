//! Phase 194 — MHD no ideal: difusión ambipolar dependiente de ionización.

use gadget_ng_core::{Particle, Vec3};
use gadget_ng_mhd::{apply_ambipolar_diffusion, ionization_fraction_proxy};

fn magnetized_gas(u: f64, dust: f64) -> Particle {
    let mut p = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), u, 0.1);
    p.b_field = Vec3::new(0.0, 1.0, 0.0);
    p.dust_to_gas = dust;
    p
}

#[test]
fn ionization_proxy_tracks_thermal_state() {
    let cold = magnetized_gas(0.01, 0.0);
    let hot = magnetized_gas(10.0, 0.0);
    assert!(
        ionization_fraction_proxy(&hot, 1e-4, 1.0) > ionization_fraction_proxy(&cold, 1e-4, 1.0)
    );
}

#[test]
fn ambipolar_diffusion_stronger_in_dusty_neutral_gas() {
    let mut clean = vec![magnetized_gas(0.05, 0.0)];
    let mut dusty = vec![magnetized_gas(0.05, 0.02)];

    apply_ambipolar_diffusion(&mut clean, 0.05, 1e-4, 1.0, 5.0 / 3.0, 1.0);
    apply_ambipolar_diffusion(&mut dusty, 0.05, 1e-4, 1.0, 5.0 / 3.0, 1.0);

    assert!(dusty[0].b_field.norm() < clean[0].b_field.norm());
    assert!(dusty[0].internal_energy > clean[0].internal_energy);
}
