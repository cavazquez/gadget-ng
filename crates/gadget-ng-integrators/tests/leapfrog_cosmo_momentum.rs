//! Conservación aproximada del momento total en paso cosmológico (mejora 7).
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};

fn total_momentum(particles: &[Particle]) -> Vec3 {
    particles
        .iter()
        .fold(Vec3::zero(), |acc, p| acc + p.velocity * p.mass)
}

#[test]
fn lattice_symmetric_momentum_near_zero_after_cosmo_step() {
    let n = 4usize;
    let box_size = 1.0_f64;
    let dx = box_size / n as f64;
    let mass = 1.0 / (n * n * n) as f64;
    let mut parts: Vec<Particle> = Vec::new();
    let mut gid = 0usize;
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let x = (ix as f64 + 0.5) * dx;
                let y = (iy as f64 + 0.5) * dx;
                let z = (iz as f64 + 0.5) * dx;
                parts.push(Particle::new(
                    gid,
                    mass,
                    Vec3::new(x, y, z),
                    Vec3::zero(),
                ));
                gid += 1;
            }
        }
    }
    let p0 = total_momentum(&parts);
    let cf = CosmoFactors {
        drift: 1e-4,
        kick_half: 1e-4,
        kick_half2: 1e-4,
    };
    for p in parts.iter_mut() {
        p.acceleration = Vec3::zero();
    }
    let mut scratch = vec![Vec3::zero(); parts.len()];
    leapfrog_cosmo_kdk_step(&mut parts, cf, &mut scratch, |_ps, out| {
        for o in out.iter_mut() {
            *o = Vec3::zero();
        }
    });
    let p1 = total_momentum(&parts);
    let diff = (p1 - p0).norm();
    assert!(
        diff < 1e-12,
        "Δ|Σ m v| demasiado grande para red simétrica: {diff:.3e}"
    );
}
