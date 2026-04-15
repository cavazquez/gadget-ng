use crate::config::{IcKind, RunConfig};
use crate::particle::Particle;
use crate::vec3::Vec3;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IcError {
    #[error("particle_count must be > 0")]
    ZeroParticles,
    #[error("two_body initial conditions require particle_count == 2")]
    TwoBodyCount,
    #[error("lattice ic requires a perfect cube particle_count = n^3, got {0}")]
    LatticeNotCube(usize),
}

fn hash_u64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

fn perturb(seed: u64, gid: usize, axis: u8) -> f64 {
    let h = hash_u64(seed.wrapping_add(gid as u64).wrapping_add(axis as u64));
    let u = (h as f64) / (u64::MAX as f64);
    (u - 0.5) * 1.0e-6
}

/// Condiciones iniciales deterministas por índice global (reproducibles MPI/serial).
pub fn build_particles(cfg: &RunConfig) -> Result<Vec<Particle>, IcError> {
    let n = cfg.simulation.particle_count;
    build_particles_for_gid_range(cfg, 0, n)
}

/// Partículas con `global_id ∈ [lo, hi)` (medio abierto), orden creciente por `global_id`.
pub fn build_particles_for_gid_range(
    cfg: &RunConfig,
    lo: usize,
    hi: usize,
) -> Result<Vec<Particle>, IcError> {
    let n = cfg.simulation.particle_count;
    if n == 0 {
        return Err(IcError::ZeroParticles);
    }
    if lo > hi || hi > n {
        return Ok(Vec::new());
    }
    let seed = cfg.simulation.seed;
    let box_size = cfg.simulation.box_size;
    let g = cfg.simulation.gravitational_constant;

    match cfg.initial_conditions.kind {
        IcKind::Lattice => {
            let side = (n as f64).cbrt().round() as usize;
            if side * side * side != n {
                return Err(IcError::LatticeNotCube(n));
            }
            let mut out = Vec::new();
            let spacing = box_size / side as f64;
            for gid in lo..hi {
                let ix = gid / (side * side);
                let rem = gid % (side * side);
                let iy = rem / side;
                let iz = rem % side;
                let x = (ix as f64 + 0.5) * spacing + perturb(seed, gid, 0);
                let y = (iy as f64 + 0.5) * spacing + perturb(seed, gid, 1);
                let z = (iz as f64 + 0.5) * spacing + perturb(seed, gid, 2);
                let pos = Vec3::new(x, y, z);
                let mass = 1.0 / n as f64;
                out.push(Particle::new(gid, mass, pos, Vec3::zero()));
            }
            let _ = g;
            Ok(out)
        }
        IcKind::TwoBody {
            mass1,
            mass2,
            separation,
        } => {
            if n != 2 {
                return Err(IcError::TwoBodyCount);
            }
            let eps = cfg.simulation.softening;
            let half = separation * 0.5;
            let m1 = mass1;
            let m2 = mass2;
            let x1 = Vec3::new(-half, 0.0, 0.0);
            let x2 = Vec3::new(half, 0.0, 0.0);
            let com = (m1 * x1 + m2 * x2) / (m1 + m2);
            let p1 = x1 - com;
            let p2 = x2 - com;
            let d = separation;
            let d2_eps = d * d + eps * eps;
            let v_rel = (g * (m1 + m2) * d * d / d2_eps.powf(1.5)).sqrt();
            let v1 = Vec3::new(0.0, v_rel * (m2 / (m1 + m2)), 0.0);
            let v2 = Vec3::new(0.0, -v_rel * (m1 / (m1 + m2)), 0.0);
            let mut full = vec![Particle::new(0, m1, p1, v1), Particle::new(1, m2, p2, v2)];
            full.retain(|p| p.global_id >= lo && p.global_id < hi);
            Ok(full)
        }
    }
}
