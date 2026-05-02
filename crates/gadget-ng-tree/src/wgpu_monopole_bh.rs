//! Barnes–Hut en **wgpu**: árbol en CPU, walk FMM en GPU ([`GpuBarnesHutFmm`]: órdenes 1–3).
//!
//! El binario solo activa esta ruta con `multipole_order ≤ 3`; orden 4 requiere CPU.

use crate::octree::Octree;
use gadget_ng_core::{GravitySolver, MacSoftening, Vec3};
use gadget_ng_gpu::{BhFmmKernelParams, GpuBarnesHutFmm};

/// Solver N-body con kernel FMM en GPU (hasta orden multipolar 3).
#[derive(Clone)]
pub struct WgpuBarnesHutGpu {
    gpu: GpuBarnesHutFmm,
    theta: f64,
    multipole_order: u8,
    use_relative_criterion: bool,
    err_tol_force_acc: f64,
    softened_multipoles: bool,
    mac_softening: MacSoftening,
}

/// Alias retrocompatible (solo monopolo era el nombre histórico).
pub type WgpuMonopoleBarnesHut = WgpuBarnesHutGpu;

impl WgpuBarnesHutGpu {
    /// `multipole_order` debe ser ≤ 3 para GPU; si no, devuelve `None`.
    pub fn try_new(
        theta: f64,
        multipole_order: u8,
        use_relative_criterion: bool,
        err_tol_force_acc: f64,
        softened_multipoles: bool,
        mac_softening: MacSoftening,
    ) -> Option<Self> {
        if multipole_order == 0 || multipole_order > 3 {
            return None;
        }
        Some(Self {
            gpu: GpuBarnesHutFmm::try_new()?,
            theta,
            multipole_order,
            use_relative_criterion,
            err_tol_force_acc,
            softened_multipoles,
            mac_softening,
        })
    }

    fn kernel_params(&self, eps2: f32, g: f32) -> BhFmmKernelParams {
        let mac_u = match self.mac_softening {
            MacSoftening::Bare => 0u32,
            MacSoftening::Consistent => 1u32,
        };
        BhFmmKernelParams {
            eps2,
            g,
            theta: self.theta as f32,
            err_tol: self.err_tol_force_acc as f32,
            multipole_order: u32::from(self.multipole_order),
            use_relative_criterion: self.use_relative_criterion,
            softened_multipoles: self.softened_multipoles,
            mac_softening: mac_u,
        }
    }
}

impl GravitySolver for WgpuBarnesHutGpu {
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
        if global_positions.is_empty() || global_indices.is_empty() {
            return;
        }

        let tree = Octree::build(global_positions, global_masses);
        let nodes = tree.export_bh_fmm_gpu_nodes();
        let root = tree.root;

        let positions_f32: Vec<f32> = global_positions
            .iter()
            .flat_map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();
        let masses_f32: Vec<f32> = global_masses.iter().map(|&m| m as f32).collect();
        let query_idx: Vec<u32> = global_indices.iter().map(|&i| i as u32).collect();

        let kp = self.kernel_params(eps2 as f32, g as f32);
        let raw = self.gpu.compute_accelerations_raw(
            &positions_f32,
            &masses_f32,
            &nodes,
            root,
            &query_idx,
            kp,
        );

        for (k, chunk) in raw.chunks_exact(3).enumerate() {
            out[k] = Vec3::new(
                f64::from(chunk[0]),
                f64::from(chunk[1]),
                f64::from(chunk[2]),
            );
        }
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::BarnesHutGravity;

    #[test]
    fn wgpu_fmm_matches_cpu_order3_geometric() {
        let Some(wgpu_bh) =
            WgpuBarnesHutGpu::try_new(0.45, 3, false, 0.005, false, MacSoftening::Bare)
        else {
            eprintln!("skip: no wgpu adapter");
            return;
        };
        let cpu_bh = BarnesHutGravity {
            theta: 0.45,
            multipole_order: 3,
            use_relative_criterion: false,
            err_tol_force_acc: 0.005,
            softened_multipoles: false,
            mac_softening: MacSoftening::Bare,
        };

        let n = 20usize;
        let positions: Vec<Vec3> = (0..n)
            .map(|i| {
                let t = i as f64 * 0.31;
                Vec3::new(t.sin(), t.cos() * 0.9, (t * 0.7).sin())
            })
            .collect();
        let masses: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64 * 0.03)).collect();
        let eps2 = 0.02_f64;
        let g = 1.0_f64;
        let idx: Vec<usize> = (0..n).collect();

        let mut out_gpu = vec![Vec3::zero(); n];
        let mut out_cpu = vec![Vec3::zero(); n];

        wgpu_bh.accelerations_for_indices(&positions, &masses, eps2, g, &idx, &mut out_gpu);
        cpu_bh.accelerations_for_indices(&positions, &masses, eps2, g, &idx, &mut out_cpu);

        let mut max_rel = 0.0_f64;
        for i in 0..n {
            let d = out_gpu[i] - out_cpu[i];
            let err = d.norm();
            let scale = out_cpu[i].norm().max(1e-12);
            max_rel = max_rel.max(err / scale);
        }
        assert!(max_rel < 2e-2, "Wgpu FMM o=3 vs CPU: max_rel {max_rel:.4e}");
    }
}
