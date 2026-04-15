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
pub mod cic;
pub mod fft_poisson;
pub mod solver;

pub use solver::PmSolver;
