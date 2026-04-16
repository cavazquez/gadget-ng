//! Integrador leapfrog KDK para partículas SPH.
//!
//! El paso de integración para gas combina:
//! - Gravedad externa `particle.acceleration` (calculada por el solver gravitatorio).
//! - Aceleración hidrodinámica SPH `gas.acc_sph`.
//! - Energía interna integrada por Euler explícito: `u += du_dt · dt`.

use crate::density::compute_density;
use crate::forces::compute_sph_forces;
use crate::particle::SphParticle;
use gadget_ng_core::Vec3;

/// Un paso leapfrog KDK completo (Kick-Drift-Kick) para un sistema SPH.
///
/// # Parámetros
/// - `particles`: lista de partículas (DM + gas mezcladas).
/// - `dt`: paso de tiempo.
/// - `gravity_accel`: función que calcula la aceleración gravitatoria y la escribe
///   en `particle.acceleration` para todas las partículas (DM y gas).
pub fn sph_kdk_step<F>(particles: &mut [SphParticle], dt: f64, gravity_accel: F)
where
    F: Fn(&mut [SphParticle]),
{
    let dt2 = 0.5 * dt;

    // ── Kick 1 (½ dt) ────────────────────────────────────────────────────────
    for p in particles.iter_mut() {
        let total_acc = total_acceleration(p);
        p.velocity += total_acc * dt2;
        if let Some(gas) = p.gas.as_mut() {
            gas.u = (gas.u + gas.du_dt * dt2).max(0.0);
        }
    }

    // ── Drift (dt) ───────────────────────────────────────────────────────────
    for p in particles.iter_mut() {
        p.position += p.velocity * dt;
    }

    // ── Fuerzas al nuevo tiempo ───────────────────────────────────────────────
    gravity_accel(particles);
    compute_density(particles);
    compute_sph_forces(particles);

    // ── Kick 2 (½ dt) ────────────────────────────────────────────────────────
    for p in particles.iter_mut() {
        let total_acc = total_acceleration(p);
        p.velocity += total_acc * dt2;
        if let Some(gas) = p.gas.as_mut() {
            gas.u = (gas.u + gas.du_dt * dt2).max(0.0);
        }
    }
}

/// Aceleración total: gravedad + SPH (sólo para gas).
#[inline]
fn total_acceleration(p: &SphParticle) -> Vec3 {
    let mut a = p.acceleration;
    if let Some(gas) = p.gas.as_ref() {
        a += gas.acc_sph;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::density::compute_density;
    use crate::forces::compute_sph_forces;
    use gadget_ng_core::Vec3;

    fn no_gravity(_: &mut [SphParticle]) {}

    /// Un blob de gas en reposo sin gravedad externa no debe explotar
    /// en pocas iteraciones (energía interna acotada).
    #[test]
    fn gas_blob_no_gravity_bounded_energy() {
        let n_side = 3usize;
        let box_size = 3.0_f64;
        let dx = box_size / n_side as f64;
        let mut parts: Vec<SphParticle> = (0..n_side.pow(3))
            .map(|k| {
                let iz = k / (n_side * n_side);
                let iy = (k / n_side) % n_side;
                let ix = k % n_side;
                let pos = Vec3::new(
                    (ix as f64 + 0.5) * dx,
                    (iy as f64 + 0.5) * dx,
                    (iz as f64 + 0.5) * dx,
                );
                SphParticle::new_gas(k, 1.0, pos, Vec3::zero(), 1.0, 2.0 * dx)
            })
            .collect();

        compute_density(&mut parts);
        compute_sph_forces(&mut parts);

        let dt = 1e-3_f64;
        for _ in 0..10 {
            sph_kdk_step(&mut parts, dt, no_gravity);
        }

        let total_u: f64 = parts
            .iter()
            .filter_map(|p| p.gas.as_ref())
            .map(|g| g.u)
            .sum();
        assert!(
            total_u > 0.0 && total_u < 1000.0,
            "energía interna total = {total_u:.4}"
        );
    }
}
