//! Solver Barnes-Hut: implementa [`gadget_ng_core::GravitySolver`].
use crate::octree::Octree;
use gadget_ng_core::{GravitySolver, Vec3};

#[derive(Debug, Clone, Copy)]
pub struct BarnesHutGravity {
    pub theta: f64,
}

impl GravitySolver for BarnesHutGravity {
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
        for (k, &gi) in global_indices.iter().enumerate() {
            let xi = global_positions[gi];
            out[k] = tree.walk_accel(xi, gi, g, eps2, self.theta, global_positions, global_masses);
        }
    }
}
