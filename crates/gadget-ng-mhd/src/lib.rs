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

pub mod anisotropic;
pub mod braginskii;
pub mod cleaning;
pub mod flux_freeze;
pub mod induction;
pub mod pressure;
pub mod reconnection;
pub mod relativistic;
pub mod stats;
pub mod turbulence;
pub mod two_fluid;

pub use anisotropic::{apply_anisotropic_conduction, beta_plasma, diffuse_cr_anisotropic};
pub use braginskii::apply_braginskii_viscosity;
pub use flux_freeze::{apply_flux_freeze, flux_freeze_error, mean_gas_density};
pub use reconnection::{apply_magnetic_reconnection, sweet_parker_rate};
pub use relativistic::{advance_srmhd, em_energy_density, inject_relativistic_jet, lorentz_factor, srmhd_conserved_to_primitive, C_LIGHT};
pub use turbulence::{apply_turbulent_forcing, turbulence_stats};
pub use two_fluid::{apply_electron_ion_coupling, mean_te_over_ti};
pub use stats::{b_field_stats, magnetic_power_spectrum};
pub use cleaning::dedner_cleaning_step;
pub use induction::{advance_induction, alfven_dt, apply_artificial_resistivity, init_b_field};
pub use pressure::{apply_magnetic_forces, magnetic_pressure, maxwell_stress};

/// Permeabilidad magnética del vacío en unidades internas (adimensionalizada a 1).
pub const MU0: f64 = 1.0;
