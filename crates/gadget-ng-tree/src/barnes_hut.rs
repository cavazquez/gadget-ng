//! Solver Barnes-Hut: implementa [`gadget_ng_core::GravitySolver`].
use crate::octree::Octree;
use gadget_ng_core::{GravitySolver, MacSoftening, Vec3};

#[derive(Debug, Clone, Copy)]
pub struct BarnesHutGravity {
    pub theta: f64,
    /// Orden de expansión multipolar: 1=monopolo, 2=mono+quad, 3=mono+quad+oct.
    pub multipole_order: u8,
    /// `true` → criterio de apertura relativo (GADGET-4 `ErrTolForceAcc`).
    /// `false` → criterio geométrico clásico `s/d < theta`.
    pub use_relative_criterion: bool,
    /// Tolerancia para el criterio relativo (ignorada en modo geométrico).
    pub err_tol_force_acc: f64,
    /// `true` → aplica softening Plummer consistente en términos cuadrupolar y octupolar
    /// (reemplaza `r²` por `r² + ε²` en los denominadores, coherente con el monopolo).
    /// `false` (default) → términos bare sin suavizado (compatibilidad hacia atrás).
    pub softened_multipoles: bool,
    /// Softening aplicado al estimador del MAC relativo (ver `MacSoftening`).
    /// Sólo afecta al criterio de apertura cuando `use_relative_criterion = true`.
    pub mac_softening: MacSoftening,
}

impl Default for BarnesHutGravity {
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
            out[k] = tree.walk_accel_multipole(
                xi,
                gi,
                g,
                eps2,
                self.theta,
                global_positions,
                global_masses,
                self.multipole_order,
                self.use_relative_criterion,
                self.err_tol_force_acc,
                self.softened_multipoles,
                self.mac_softening,
            );
        }
    }
}
