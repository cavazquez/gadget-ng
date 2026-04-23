//! Análisis in-situ para gadget-ng.
//!
//! # Módulos
//! - [`fof`]: Friends-of-Friends halo finder (cell-linked-list + Union-Find).
//! - [`power_spectrum`]: estimador de P(k) via CIC + FFT 3D.
//! - [`catalog`]: escritura/lectura de catálogos en JSONL; función `analyse`.
//!
//! # Uso rápido
//!
//! ```rust,no_run
//! use gadget_ng_analysis::catalog::{analyse, AnalysisParams, write_halo_catalog, write_power_spectrum};
//!
//! // `particles` es un &[gadget_ng_core::Particle]
//! # let particles = &[];
//! let params = AnalysisParams {
//!     box_size: 100.0,
//!     b: 0.2,
//!     min_particles: 20,
//!     pk_mesh: 64,
//!     ..Default::default()
//! };
//! let result = analyse(particles, &params);
//! println!("{} halos encontrados", result.halos.len());
//! ```

pub mod catalog;
pub mod correlation;
pub mod fof;
pub mod halo_mass_function;
pub mod halofit;
pub mod nfw;
pub mod pk_correction;
pub mod power_spectrum;

pub use catalog::{
    analyse, read_halo_catalog, read_power_spectrum, write_halo_catalog, write_power_spectrum,
    AnalysisParams, AnalysisResult,
};
pub use correlation::{two_point_correlation_fft, two_point_correlation_pairs, XiBin};
pub use fof::FofHalo;
pub use halo_mass_function::{
    hmf_press_schechter, hmf_sheth_tormen, lagrange_radius, mass_function_table, multiplicity_ps,
    multiplicity_st, sigma_m, total_halo_density, HmfBin, HmfParams, DELTA_C, RHO_CRIT_H2,
};
pub use halofit::{halofit_pk, p_linear_eh, HalofitCosmo};
pub use nfw::{
    concentration_bhattacharya2013, concentration_duffy2008, concentration_ludlow2016,
    fit_nfw_concentration, measure_density_profile, r200_from_m200, rho_crit_z, DensityBin,
    NfwFitResult, NfwProfile, DELTA_VIRIALIZED, RHO_CRIT0,
};
pub use pk_correction::{a_grid, correct_pk, correct_pk_with_shot_noise, measure_rn, RnModel};
pub use power_spectrum::PkBin;
