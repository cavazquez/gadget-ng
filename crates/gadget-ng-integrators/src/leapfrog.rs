use gadget_ng_core::{Particle, Vec3};

/// Un paso leapfrog en forma kick–drift–kick (KDK) con paso `dt` fijo.
/// `compute` debe escribir aceleraciones alineadas con `particles` (mismo orden).
pub fn leapfrog_kdk_step(
    particles: &mut [Particle],
    dt: f64,
    scratch_acc: &mut [Vec3],
    mut compute: impl FnMut(&[Particle], &mut [Vec3]),
) {
    assert_eq!(particles.len(), scratch_acc.len());
    compute(particles, scratch_acc);
    for (p, &a) in particles.iter_mut().zip(scratch_acc.iter()) {
        p.velocity += a * (0.5 * dt);
    }
    for p in particles.iter_mut() {
        p.position += p.velocity * dt;
    }
    compute(particles, scratch_acc);
    for (p, &a) in particles.iter_mut().zip(scratch_acc.iter()) {
        p.velocity += a * (0.5 * dt);
        p.acceleration = a;
    }
}
