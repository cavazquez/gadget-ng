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
pub mod cic;
pub mod distributed;
pub mod fft_poisson;
pub mod pencil_fft;
pub mod slab_fft;
pub mod slab_pm;
pub mod solver;

pub use amr::{
    amr_pm_accels, amr_pm_accels_with_stats, AmrParams, AmrStats, PatchGrid,
};
pub use pencil_fft::{solve_forces_pencil2d, PencilLayout2D};
pub use slab_fft::SlabLayout;
pub use solver::PmSolver;
