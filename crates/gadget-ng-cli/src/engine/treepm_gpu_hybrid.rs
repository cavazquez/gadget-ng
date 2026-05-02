//! TreePM híbrido: largo alcance con PM CUDA filtrado (`r_split`) + corto alcance [`GpuTreePmShortRange`].

use gadget_ng_core::{GravitySolver, Vec3};
use gadget_ng_cuda::CudaPmSolver;
use gadget_ng_gpu::GpuTreePmShortRange;
use gadget_ng_tree::Octree;

/// Igual semántica que [`gadget_ng_treepm::TreePmSolver`], con LR/SR en GPU cuando CUDA+wgpu están disponibles.
pub(crate) struct TreePmGpuHybridSolver {
    pm: CudaPmSolver,
    sr: GpuTreePmShortRange,
    r_split_effective: f64,
}

impl TreePmGpuHybridSolver {
    pub fn try_from_config(pm_grid_size: usize, box_size: f64, r_split_cfg: f64) -> Option<Self> {
        let r_s = if r_split_cfg > 0.0 {
            r_split_cfg
        } else {
            2.5 * box_size / pm_grid_size as f64
        };
        let pm = CudaPmSolver::try_new_with_r_split(pm_grid_size, box_size, r_s)?;
        let sr = GpuTreePmShortRange::try_new()?;
        Some(Self {
            pm,
            sr,
            r_split_effective: r_s,
        })
    }
}

impl GravitySolver for TreePmGpuHybridSolver {
    fn accelerations_for_indices(
        &self,
        global_positions: &[Vec3],
        global_masses: &[f64],
        eps2: f64,
        g: f64,
        global_indices: &[usize],
        out: &mut [Vec3],
    ) {
        assert_eq!(global_indices.len(), out.len());
        if global_indices.is_empty() {
            return;
        }

        let r_s = self.r_split_effective;
        let r_cut = 5.0 * r_s;
        let r_cut2 = r_cut * r_cut;

        let mut acc_lr = vec![Vec3::zero(); global_indices.len()];
        self.pm.accelerations_for_indices(
            global_positions,
            global_masses,
            eps2,
            g,
            global_indices,
            &mut acc_lr,
        );

        let tree = Octree::build(global_positions, global_masses);
        let nodes = tree.export_bh_monopole_gpu_nodes();
        let positions_f32: Vec<f32> = global_positions
            .iter()
            .flat_map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();
        let masses_f32: Vec<f32> = global_masses.iter().map(|&m| m as f32).collect();
        let query_idx: Vec<u32> = global_indices.iter().map(|&i| i as u32).collect();

        let sr_raw = self.sr.compute_accelerations_raw(
            &positions_f32,
            &masses_f32,
            &nodes,
            tree.root,
            &query_idx,
            eps2 as f32,
            g as f32,
            r_s as f32,
            r_cut2 as f32,
        );

        for k in 0..out.len() {
            out[k] = acc_lr[k]
                + Vec3::new(
                    sr_raw[3 * k] as f64,
                    sr_raw[3 * k + 1] as f64,
                    sr_raw[3 * k + 2] as f64,
                );
        }
    }
}
