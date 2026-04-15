use crate::particle::Particle;
use crate::vec3::Vec3;

/// Aceleración gravitatoria pareada con suavizado Plummer: denominador (r² + ε²)^{3/2}.
pub fn pairwise_accel_plummer(pos_i: Vec3, mass_j: f64, pos_j: Vec3, g: f64, eps2: f64) -> Vec3 {
    // Aceleración sobre i debida a j: G m_j (x_j - x_i) / (|x_i-x_j|^2 + eps^2)^{3/2}
    let r = pos_j - pos_i;
    let r2 = r.dot(r) + eps2;
    let inv = 1.0 / r2.sqrt();
    let inv3 = inv * inv * inv;
    g * mass_j * r * inv3
}

pub trait GravitySolver: Send + Sync {
    /// Calcula aceleraciones para un subconjunto de índices globales usando estado global.
    fn accelerations_for_indices(
        &self,
        global_positions: &[Vec3],
        global_masses: &[f64],
        eps2: f64,
        g: f64,
        global_indices: &[usize],
        out: &mut [Vec3],
    );
}

/// Gravedad directa O(N²) globalmente consistente: mismo orden de suma j=0..N-1 que serial.
pub struct DirectGravity;

impl GravitySolver for DirectGravity {
    fn accelerations_for_indices(
        &self,
        global_positions: &[Vec3],
        global_masses: &[f64],
        eps2: f64,
        g: f64,
        global_indices: &[usize],
        out: &mut [Vec3],
    ) {
        assert_eq!(global_positions.len(), global_masses.len());
        assert_eq!(global_indices.len(), out.len());
        for (k, &gi) in global_indices.iter().enumerate() {
            let xi = global_positions[gi];
            let mut a = Vec3::zero();
            for j in 0..global_positions.len() {
                if j == gi {
                    continue;
                }
                a += pairwise_accel_plummer(xi, global_masses[j], global_positions[j], g, eps2);
            }
            out[k] = a;
        }
    }
}

/// Variante serial: todas las partículas en un slice.
pub fn accelerations_all_particles(
    solver: &impl GravitySolver,
    particles: &[Particle],
    eps2: f64,
    g_const: f64,
    scratch_acc: &mut [Vec3],
) {
    let n = particles.len();
    let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let mass: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    let idx: Vec<usize> = (0..n).collect();
    solver.accelerations_for_indices(&pos, &mass, eps2, g_const, &idx, scratch_acc);
    debug_assert_eq!(scratch_acc.len(), n);
}

#[cfg(feature = "simd")]
pub mod parallel_direct {
    use super::*;
    use rayon::prelude::*;

    pub fn accelerations_all_particles_rayon(
        solver: &DirectGravity,
        particles: &[Particle],
        eps2: f64,
        g_const: f64,
        scratch_acc: &mut [Vec3],
    ) {
        let n = particles.len();
        let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
        let mass: Vec<f64> = particles.iter().map(|p| p.mass).collect();
        scratch_acc
            .par_iter_mut()
            .enumerate()
            .for_each(|(gi, out_a)| {
                let xi = pos[gi];
                let mut a = Vec3::zero();
                for j in 0..n {
                    if j == gi {
                        continue;
                    }
                    a += pairwise_accel_plummer(xi, mass[j], pos[j], g_const, eps2);
                }
                *out_a = a;
            });
        let _ = solver;
    }
}
