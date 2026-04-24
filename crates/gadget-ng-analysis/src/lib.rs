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

pub mod assembly_bias;
pub mod bispectrum;
pub mod pk_rsd;
pub mod catalog;
pub mod correlation;
pub mod fof;
pub mod luminosity;
#[cfg(feature = "parallel")]
pub mod fof_parallel;
pub mod halo_mass_function;
pub mod halofit;
pub mod halo_spin;
pub mod merger_tree;
pub mod nfw;
pub mod pk_correction;
pub mod power_spectrum;
pub mod subfind;
pub mod velocity_profile;

pub use catalog::{
    analyse, read_halo_catalog, read_power_spectrum, write_halo_catalog, write_power_spectrum,
    AnalysisParams, AnalysisResult,
};
pub use correlation::{two_point_correlation_fft, two_point_correlation_pairs, XiBin};
pub use fof::{find_halos_combined, find_halos_with_membership, particle_snapshots_from_catalog, FofHalo};
#[cfg(feature = "parallel")]
pub use fof_parallel::find_halos_parallel;
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
pub use merger_tree::{
    build_merger_forest, mah_main_branch, mah_mcbride2009, MassAccretionHistory, MergerForest,
    MergerTreeNode, ParticleSnapshot,
};
pub use pk_correction::{a_grid, correct_pk, correct_pk_with_shot_noise, measure_rn, RnModel};
pub use power_spectrum::PkBin;
pub use bispectrum::{
    bispectrum_equilateral, bispectrum_isosceles, reduced_bispectrum, BkBin, BkIsoscelesBin,
};
pub use assembly_bias::{
    compute_assembly_bias, spearman_correlation, AssemblyBiasParams, AssemblyBiasResult,
};
pub use pk_rsd::{
    compute_pk_multipoles, kaiser_multipole_ratios, pk_multipoles, pk_redshift_space,
    LosAxis, PkMultipoleBin, PkRsdBin, PkRsdParams,
};
pub use halo_spin::{compute_halo_spins, halo_spin, HaloSpin, SpinParams};
pub use subfind::{find_subhalos, local_density_sph, SubfindParams, SubhaloRecord};
pub use velocity_profile::{
    sigma_1d, velocity_anisotropy, velocity_profile, VelocityProfileBin, VelocityProfileParams,
};
pub use luminosity::{bv_color, galaxy_luminosity, gr_color, stellar_luminosity_solar, LuminosityResult};
