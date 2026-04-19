//! Solver **TreePM** para **gadget-ng**.
//!
//! Combina un solver Particle-Mesh (PM) con filtro Gaussiano en k-space para las
//! fuerzas de **largo alcance** con un recorrido de octree usando el kernel erfc
//! para las fuerzas de **corto alcance**.
//!
//! La suma largo + corto alcance recupera el Newton exacto:
//! `erf(r/(√2·r_s)) + erfc(r/(√2·r_s)) = 1`.
//!
//! # Configuración TOML
//!
//! ```toml
//! [gravity]
//! solver       = "tree_pm"
//! pm_grid_size = 64
//! r_split      = 0.0   # 0 → auto: 2.5 × cell_size
//! ```
pub mod distributed;
pub mod short_range;
pub mod solver;

pub use distributed::{
    halo_stats, pm_scatter_gather_accels, short_range_accels_sfc, short_range_accels_slab,
    HaloStats, PmScatterStats, SfcShortRangeParams, SlabShortRangeParams,
};
pub use solver::TreePmSolver;
