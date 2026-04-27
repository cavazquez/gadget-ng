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

    /// Como [`accelerations_for_indices`] pero también devuelve el coste de interacción
    /// estimado (p. ej. nodos abiertos del árbol) para cada partícula en `global_indices`.
    ///
    /// El vector `costs` se rellena con un valor `u64` por entrada de `global_indices`.
    /// La implementación por defecto llama a [`accelerations_for_indices`] y deja `costs`
    /// vacío (coste cero implícito), compatible con solvers que no rastrean costes.
    ///
    /// Implementaciones que soporten rastreo de coste deben rellenar `costs` con la
    /// métrica de trabajo por partícula (p. ej. `opened_nodes` del walk Barnes-Hut).
    #[allow(clippy::too_many_arguments)]
    fn accelerations_with_costs(
        &self,
        global_positions: &[Vec3],
        global_masses: &[f64],
        eps2: f64,
        g: f64,
        global_indices: &[usize],
        out: &mut [Vec3],
        costs: &mut Vec<u64>,
    ) {
        costs.clear();
        self.accelerations_for_indices(
            global_positions,
            global_masses,
            eps2,
            g,
            global_indices,
            out,
        );
    }
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

/// Gravedad directa O(N²) paralelizada con Rayon (feature `simd`).
///
/// El bucle externo (sobre i) se distribuye en el pool de Rayon; el bucle interno
/// usa el kernel SoA+caché-blocking+AVX2 de `gravity_simd::accel_soa_blocked`.
/// **No determinista** respecto al orden de suma entre partículas; no garantiza
/// paridad bit-a-bit con el modo serial ni con `MpiRuntime`.
#[cfg(feature = "simd")]
pub struct RayonDirectGravity;

#[cfg(feature = "simd")]
impl GravitySolver for RayonDirectGravity {
    fn accelerations_for_indices(
        &self,
        global_positions: &[Vec3],
        global_masses: &[f64],
        eps2: f64,
        g: f64,
        global_indices: &[usize],
        out: &mut [Vec3],
    ) {
        use crate::gravity_simd::{KernelParams, accel_soa_blocked};
        use rayon::prelude::*;

        assert_eq!(global_positions.len(), global_masses.len());
        assert_eq!(global_indices.len(), out.len());

        // Extracción SoA una sola vez, fuera del par_iter_mut.
        let xs: Vec<f64> = global_positions.iter().map(|p| p.x).collect();
        let ys: Vec<f64> = global_positions.iter().map(|p| p.y).collect();
        let zs: Vec<f64> = global_positions.iter().map(|p| p.z).collect();
        let params = KernelParams {
            xs: &xs,
            ys: &ys,
            zs: &zs,
            masses: global_masses,
            eps2,
            g,
        };

        out.par_iter_mut()
            .zip(global_indices.par_iter())
            .for_each(|(a, &gi)| {
                let (ax, ay, az) = accel_soa_blocked(xs[gi], ys[gi], zs[gi], gi, &params);
                *a = Vec3::new(ax, ay, az);
            });
    }
}
