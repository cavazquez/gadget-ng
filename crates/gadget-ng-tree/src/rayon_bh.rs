//! Solver Barnes-Hut paralelizado con Rayon (feature `simd`).
//!
//! El octree se construye una vez por llamada y es de solo lectura durante el walk,
//! lo que permite que múltiples hilos ejecuten `walk_accel` simultáneamente de forma segura.
//!
//! **No determinista** respecto al orden de suma: no garantiza paridad bit-a-bit con
//! el solver serial ni con `MpiRuntime`.
use rayon::prelude::*;

use crate::octree::Octree;
use gadget_ng_core::{GravitySolver, Vec3};

/// Solver Barnes-Hut con paralelismo Rayon en el bucle de partículas.
#[derive(Debug, Clone, Copy)]
pub struct RayonBarnesHutGravity {
    pub theta: f64,
}

impl GravitySolver for RayonBarnesHutGravity {
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
        if global_positions.is_empty() {
            return;
        }
        let tree = Octree::build(global_positions, global_masses);
        let theta = self.theta;
        out.par_iter_mut()
            .zip(global_indices.par_iter())
            .for_each(|(a, &gi)| {
                *a = tree.walk_accel(
                    global_positions[gi],
                    gi,
                    g,
                    eps2,
                    theta,
                    global_positions,
                    global_masses,
                );
            });
    }
}
