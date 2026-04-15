//! Oscilador armónico isótropo: energía discreta acotada con KDK y paso pequeño.
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_integrators::leapfrog_kdk_step;

#[test]
fn harmonic_energy_bounded() {
    let m = 1.0_f64;
    let k = 1.0_f64;
    let dt = 0.01_f64;
    let mut p = vec![Particle::new(
        0,
        m,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.2, 0.0),
    )];
    let mut scratch = vec![Vec3::zero(); 1];
    let energy = |parts: &[Particle]| -> f64 {
        let x = parts[0].position;
        let v = parts[0].velocity;
        0.5 * m * v.dot(v) + 0.5 * k * x.dot(x)
    };
    let e0 = energy(&p);
    let mut e_max = e0.abs();
    for _ in 0..500 {
        leapfrog_kdk_step(&mut p, dt, &mut scratch, |parts, acc| {
            acc[0] = -k * parts[0].position;
        });
        let e = energy(&p);
        e_max = e_max.max(e.abs());
    }
    assert!(
        e_max < 2.0 * e0.abs() + 1e-6,
        "energía máxima {e_max} frente a inicial {e0}"
    );
}
