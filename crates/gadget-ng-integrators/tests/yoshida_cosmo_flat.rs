//! La variante cosmológica de Yoshida 4º orden con `CosmoFactors::flat(w·dt)`
//! debe ser bit-a-bit idéntica a la variante Newtoniana, garantizando
//! retrocompatibilidad sin regresión numérica cuando `a(t) ≡ 1`.
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_integrators::{
    yoshida4_cosmo_kdk_step, yoshida4_kdk_step, CosmoFactors, YOSHIDA4_W0, YOSHIDA4_W1,
};

fn force_kepler(parts: &[Particle], acc: &mut [Vec3]) {
    let r = parts[0].position;
    let r3 = r.norm().powi(3);
    acc[0] = r * (-1.0 / r3);
}

#[test]
fn yoshida4_cosmo_flat_bitexact_newtonian() {
    let make_particle = || {
        Particle::new(0, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.8, 0.0))
    };
    let dt = 0.03_f64;
    let cf = [
        CosmoFactors::flat(YOSHIDA4_W1 * dt),
        CosmoFactors::flat(YOSHIDA4_W0 * dt),
        CosmoFactors::flat(YOSHIDA4_W1 * dt),
    ];

    let mut p1 = vec![make_particle()];
    let mut p2 = vec![make_particle()];
    let mut s1 = vec![Vec3::zero()];
    let mut s2 = vec![Vec3::zero()];

    for _ in 0..25 {
        yoshida4_kdk_step(&mut p1, dt, &mut s1, force_kepler);
        yoshida4_cosmo_kdk_step(&mut p2, cf, &mut s2, force_kepler);
    }

    assert_eq!(
        p1[0].position, p2[0].position,
        "posición bit-exacta requerida"
    );
    assert_eq!(
        p1[0].velocity, p2[0].velocity,
        "velocidad bit-exacta requerida"
    );
    assert_eq!(p1[0].acceleration, p2[0].acceleration);
}
