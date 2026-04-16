//! Puente entre `gadget-ng-gpu` y los tipos de `gadget-ng-core`.
//!
//! Este módulo existe para evitar la dependencia circular:
//! - `gadget-ng-gpu` no depende de `gadget-ng-core`.
//! - `gadget-ng-core` (con `feature = "gpu"`) depende de `gadget-ng-gpu`.
//! - Las impls de traits definidos en core para tipos de gpu viven aquí.
use gadget_ng_gpu::GpuDirectGravity;

use crate::gravity::GravitySolver;
use crate::particle::Particle;
use crate::vec3::Vec3;

// ── Conversiones GpuParticlesSoA ↔ Particle ──────────────────────────────────

/// Extensión de `GpuParticlesSoA` para convertir hacia/desde `Particle`.
///
/// Estas impl viven aquí (y no en `gadget-ng-gpu`) porque `Particle` está en
/// `gadget-ng-core` y ese crate no puede depender de este crate sin ciclo.
pub trait GpuParticlesSoAExt {
    fn from_particles(particles: &[Particle]) -> Self;
    fn into_particles(self) -> Vec<Particle>;
}

impl GpuParticlesSoAExt for gadget_ng_gpu::GpuParticlesSoA {
    fn from_particles(particles: &[Particle]) -> Self {
        let n = particles.len();
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut zs = Vec::with_capacity(n);
        let mut vxs = Vec::with_capacity(n);
        let mut vys = Vec::with_capacity(n);
        let mut vzs = Vec::with_capacity(n);
        let mut masses = Vec::with_capacity(n);
        let mut ids = Vec::with_capacity(n);
        for p in particles {
            xs.push(p.position.x);
            ys.push(p.position.y);
            zs.push(p.position.z);
            vxs.push(p.velocity.x);
            vys.push(p.velocity.y);
            vzs.push(p.velocity.z);
            masses.push(p.mass);
            ids.push(p.global_id);
        }
        gadget_ng_gpu::GpuParticlesSoA::from_arrays(xs, ys, zs, vxs, vys, vzs, masses, ids)
    }

    fn into_particles(self) -> Vec<Particle> {
        let n = self.ids.len();
        (0..n)
            .map(|i| {
                Particle::new(
                    self.ids[i],
                    self.masses[i],
                    Vec3::new(self.xs[i], self.ys[i], self.zs[i]),
                    Vec3::new(self.vxs[i], self.vys[i], self.vzs[i]),
                )
            })
            .collect()
    }
}

// ── GravitySolver para GpuDirectGravity ──────────────────────────────────────

impl GravitySolver for GpuDirectGravity {
    /// Calcula aceleraciones usando el compute shader wgpu.
    ///
    /// Los datos f64 se convierten a f32 para la transferencia al device GPU.
    /// El error relativo introducido es O(1e-7) (precisión f32).
    fn accelerations_for_indices(
        &self,
        global_positions: &[Vec3],
        global_masses:    &[f64],
        eps2:             f64,
        g:                f64,
        global_indices:   &[usize],
        out:              &mut [Vec3],
    ) {
        assert_eq!(global_indices.len(), out.len());
        if global_indices.is_empty() {
            return;
        }

        // Empaquetar posiciones globales como [x0,y0,z0, x1,y1,z1, ...] en f32
        let positions_f32: Vec<f32> = global_positions
            .iter()
            .flat_map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();
        let masses_f32: Vec<f32> = global_masses.iter().map(|&m| m as f32).collect();
        let query_idx: Vec<u32> = global_indices.iter().map(|&i| i as u32).collect();

        let raw = self.compute_accelerations_raw(
            &positions_f32,
            &masses_f32,
            &query_idx,
            eps2 as f32,
            g as f32,
        );

        // Convertir resultados f32 → Vec3 f64
        for (k, chunk) in raw.chunks_exact(3).enumerate() {
            out[k] = Vec3::new(chunk[0] as f64, chunk[1] as f64, chunk[2] as f64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::Vec3;

    #[test]
    fn soa_particle_roundtrip() {
        use crate::gpu_bridge::GpuParticlesSoAExt;
        let particles = vec![
            Particle::new(0, 1.0, Vec3::new(0.1, 0.2, 0.3), Vec3::new(1.0, 2.0, 3.0)),
            Particle::new(1, 2.0, Vec3::new(4.0, 5.0, 6.0), Vec3::new(-1.0, 0.0, 1.0)),
        ];
        let soa = gadget_ng_gpu::GpuParticlesSoA::from_particles(&particles);
        assert_eq!(soa.len(), 2);
        assert!((soa.xs[0] - 0.1).abs() < 1e-15);
        let restored = soa.into_particles();
        assert_eq!(restored[0], particles[0]);
        assert_eq!(restored[1], particles[1]);
    }
}
