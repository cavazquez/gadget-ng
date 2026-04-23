//! Solver Particle-Mesh (PM) para **gadget-ng**.
//!
//! Implementa [`PmSolver`] que resuelve la ecuación de Poisson gravitacional en
//! k-space usando FFT 3D periódica. Las fuerzas se obtienen con interpolación
//! Cloud-in-Cell (CIC) entre el grid y las posiciones de las partículas.
//!
//! # Uso en TOML
//!
//! ```toml
//! [gravity]
//! solver = "pm"
//! pm_grid_size = 64   # NM: grid NM³, potencia de 2 recomendada
//! ```
pub mod amr;
pub mod amr_mpi;
pub mod cic;
pub mod distributed;
pub mod fft_poisson;
pub mod pencil_fft;
pub mod slab_fft;
pub mod slab_pm;
pub mod solver;

pub use amr::{
    amr_pm_accels, amr_pm_accels_multilevel, amr_pm_accels_multilevel_with_stats,
    amr_pm_accels_with_stats, build_amr_hierarchy, identify_refinement_patches,
    AmrLevel, AmrMultilevelStats, AmrParams, AmrStats, PatchGrid,
};
pub use amr_mpi::{
    amr_pm_accels_multilevel_mpi, broadcast_patch_forces, build_amr_hierarchy_mpi,
    AmrPatchMessage, AmrRuntime,
};
#[cfg(feature = "mpi")]
pub use amr_mpi::{
    amr_pm_accels_multilevel_mpi_real, broadcast_patch_forces_mpi, build_amr_hierarchy_mpi_real,
};
pub use pencil_fft::{solve_forces_pencil2d, PencilLayout2D};
pub use slab_fft::SlabLayout;
pub use solver::PmSolver;
