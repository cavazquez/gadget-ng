//! Solver Barnes-Hut paralelizado con Rayon (feature `simd`).
//!
//! El octree se construye una vez por llamada y es de solo lectura durante el walk,
//! lo que permite que múltiples hilos ejecuten `walk_accel` simultáneamente de forma segura.
//!
//! **No determinista** respecto al orden de suma: no garantiza paridad bit-a-bit con
//! el solver serial ni con `MpiRuntime`.
//!
//! En `gadget-ng-cli`, con `feature = simd` y `[performance] deterministic = false` y
//! `solver = "Tree"`, además de este trait se paralelizan los walks del árbol en los
//! kernels MPI (slab/SFC, LET, block-timestep jerárquico) vía `par_iter` en
//! `engine/gravity.rs`, con la misma convención de no-determinismo.
use rayon::prelude::*;

use crate::octree::Octree;
use gadget_ng_core::{GravitySolver, MacSoftening, Vec3};

/// Solver Barnes-Hut con paralelismo Rayon en el bucle de partículas.
#[derive(Debug, Clone, Copy)]
pub struct RayonBarnesHutGravity {
    pub theta: f64,
    /// Orden de expansión multipolar: 1=monopolo, 2=mono+quad, 3=+oct, 4=+hexadecapolo.
    pub multipole_order: u8,
    /// `true` → criterio de apertura relativo (GADGET-4 `ErrTolForceAcc`).
    pub use_relative_criterion: bool,
    /// Tolerancia para el criterio relativo.
    pub err_tol_force_acc: f64,
    /// `true` → softening Plummer consistente en términos cuadrupolar y octupolar.
    pub softened_multipoles: bool,
    /// Softening aplicado al estimador del MAC relativo.
    pub mac_softening: MacSoftening,
}

impl Default for RayonBarnesHutGravity {
    fn default() -> Self {
        Self {
            theta: 0.5,
            multipole_order: 3,
            use_relative_criterion: false,
            err_tol_force_acc: 0.005,
            softened_multipoles: false,
            mac_softening: MacSoftening::Bare,
        }
    }
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
        let multipole_order = self.multipole_order;
        let use_relative_criterion = self.use_relative_criterion;
        let err_tol = self.err_tol_force_acc;
        let softened_multipoles = self.softened_multipoles;
        let mac_softening = self.mac_softening;
        out.par_iter_mut()
            .zip(global_indices.par_iter())
            .for_each(|(a, &gi)| {
                *a = tree.walk_accel_multipole(
                    global_positions[gi],
                    gi,
                    g,
                    eps2,
                    theta,
                    global_positions,
                    global_masses,
                    multipole_order,
                    use_relative_criterion,
                    err_tol,
                    softened_multipoles,
                    mac_softening,
                );
            });
    }
}
