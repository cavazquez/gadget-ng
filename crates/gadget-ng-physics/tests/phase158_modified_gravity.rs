//! Tests de integración — Phase 158: Gravedad modificada f(R) con screening chameleon.

use gadget_ng_core::{
    apply_modified_gravity, chameleon_field, fifth_force_factor, CosmologyParams, FRParams,
    Particle, Vec3,
};

fn dm_particle(x: f64, acc_x: f64, h: f64, m: f64) -> Particle {
    let mut p = Particle::new(0, m, Vec3 { x, y: 0.0, z: 0.0 }, Vec3::zero());
    p.acceleration = Vec3 { x: acc_x, y: 0.0, z: 0.0 };
    p.smoothing_length = h;
    p
}

// T1: f_r0=0 idéntico a GR (sin quinta fuerza)
#[test]
fn f_r0_zero_is_gr() {
    let params = FRParams { f_r0: 0.0, n: 1.0 };
    let cosmo = CosmologyParams::new(0.3, 0.7, 0.1);
    let mut particles = vec![dm_particle(0.0, 1.0, 0.5, 1.0)];
    let acc_before = particles[0].acceleration.x;
    apply_modified_gravity(&mut particles, &params, &cosmo, 1.0);
    assert_eq!(particles[0].acceleration.x, acc_before, "f_r0=0 no debe cambiar la aceleración");
}

// T2: quinta fuerza factor en [0, 1]
#[test]
fn fifth_force_factor_in_range() {
    let f_r_local = 1.0e-4;
    let f_r0 = 1.0e-4;
    let factor = fifth_force_factor(f_r_local, f_r0);
    assert!(factor >= 0.0 && factor <= 1.0, "Factor de quinta fuerza debe estar en [0,1]: {}", factor);
}

// T3: screening en región densa
#[test]
fn screening_in_dense_region() {
    let delta_rho_dense = 1000.0;
    let f_r0 = 1.0e-4;
    let n = 1.0;
    let f_r_local = chameleon_field(delta_rho_dense, f_r0, n);
    let factor = fifth_force_factor(f_r_local, f_r0);
    assert!(factor < 0.01, "En región densa, factor debe ser << 1 (screening), got {}", factor);
}

// T4: aceleración aumenta o se mantiene con f(R)
#[test]
fn acceleration_nondecreasing_with_fr() {
    let params = FRParams { f_r0: 1.0e-4, n: 1.0 };
    let cosmo = CosmologyParams::new(0.3, 0.7, 0.1);
    let mut particles = vec![dm_particle(0.0, 3.0, 100.0, 0.001)];
    let acc_before = particles[0].acceleration.x;
    apply_modified_gravity(&mut particles, &params, &cosmo, 1.0);
    let acc_after = particles[0].acceleration.x;
    assert!(acc_after >= acc_before, "La aceleración debe aumentar o mantenerse con f(R)");
}

// T5: chameleon_field decrece con delta_rho
#[test]
fn chameleon_field_decreasing() {
    let f_r0 = 1.0e-4;
    let n = 1.0;
    let f_low = chameleon_field(0.0, f_r0, n);
    let f_high = chameleon_field(10.0, f_r0, n);
    assert!(f_low >= f_high, "Campo chameleon debe decrecer con densidad");
}

// T6: N=100 sin panics, aceleraciones finitas
#[test]
fn modified_gravity_n100_no_panic() {
    let params = FRParams { f_r0: 1.0e-5, n: 1.0 };
    let cosmo = CosmologyParams::new(0.3, 0.7, 0.1);
    let mut particles: Vec<Particle> = (0..100).map(|i| {
        dm_particle(i as f64 * 0.1, 1.0, 0.3, 0.1)
    }).collect();
    apply_modified_gravity(&mut particles, &params, &cosmo, 1.0);
    for p in &particles {
        assert!(p.acceleration.x.is_finite(), "Aceleración debe ser finita");
    }
}
