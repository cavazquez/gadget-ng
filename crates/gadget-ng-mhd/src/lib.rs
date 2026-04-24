//! Magnetohydrodynamics (MHD) para gadget-ng (Phase 123+).
//!
//! ## Arquitectura
//!
//! Este crate implementa MHD ideal en la formulación SPH de Morris & Monaghan (1997)
//! extendida con el esquema de limpieza de divergencia de Dedner et al. (2002).
//!
//! ### Módulos
//!
//! - [`induction`]: Ecuación de inducción `dB/dt = ∇×(v×B)` — Phase 123.
//! - [`pressure`]: Presión magnética `P_B = |B|²/(2μ₀)` y tensor de Maxwell — Phase 124.
//! - [`cleaning`]: Esquema de Dedner `∂ψ/∂t + c_h²∇·B = -c_r ψ` — Phase 125.
//!
//! ## Referencia
//!
//! - Morris & Monaghan (1997), J. Comput. Phys. 136, 41.
//! - Dedner et al. (2002), J. Comput. Phys. 175, 645.
//! - Price & Monaghan (2005), MNRAS 364, 384.

pub mod cleaning;
pub mod induction;
pub mod pressure;

pub use cleaning::dedner_cleaning_step;
pub use induction::advance_induction;
pub use pressure::{apply_magnetic_forces, magnetic_pressure, maxwell_stress};

/// Permeabilidad magnética del vacío en unidades internas (adimensionalizada a 1).
pub const MU0: f64 = 1.0;
