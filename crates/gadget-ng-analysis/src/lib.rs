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
pub mod catalog;
pub mod correlation;
pub mod emission_lines;
pub mod fof;
#[cfg(feature = "parallel")]
pub mod fof_parallel;
pub mod halo_mass_function;
pub mod halo_spin;
pub mod halofit;
pub mod luminosity;
pub mod merger_tree;
pub mod mock_catalog;
pub mod nfw;
pub mod pk_correction;
pub mod pk_rsd;
pub mod power_spectrum;
pub mod sps_tables;
pub mod subfind;
pub mod velocity_profile;
pub mod xray;

pub use assembly_bias::{
    AssemblyBiasParams, AssemblyBiasResult, compute_assembly_bias, spearman_correlation,
};
pub use bispectrum::{
    BkBin, BkIsoscelesBin, bispectrum_equilateral, bispectrum_isosceles, reduced_bispectrum,
};
pub use catalog::{
    AnalysisParams, AnalysisResult, analyse, read_halo_catalog, read_power_spectrum,
    write_halo_catalog, write_power_spectrum,
};
pub use correlation::{XiBin, two_point_correlation_fft, two_point_correlation_pairs};
pub use emission_lines::{
    EmissionLine, bpt_diagram, compute_emission_lines, emissivity_halpha, emissivity_nii,
    emissivity_oiii,
};
pub use fof::{
    FofHalo, find_halos_combined, find_halos_with_membership, particle_snapshots_from_catalog,
};
#[cfg(feature = "parallel")]
pub use fof_parallel::find_halos_parallel;
pub use halo_mass_function::{
    DELTA_C, HmfBin, HmfParams, RHO_CRIT_H2, hmf_press_schechter, hmf_sheth_tormen,
    lagrange_radius, mass_function_table, multiplicity_ps, multiplicity_st, sigma_m,
    total_halo_density,
};
pub use halo_spin::{HaloSpin, SpinParams, compute_halo_spins, halo_spin};
pub use halofit::{HalofitCosmo, halofit_pk, p_linear_eh};
pub use luminosity::{
    LuminosityResult, SedResult, bv_color, galaxy_luminosity, gr_color, stellar_luminosity_solar,
};
pub use merger_tree::{
    MassAccretionHistory, MergerForest, MergerTreeNode, ParticleSnapshot, build_merger_forest,
    mah_main_branch, mah_mcbride2009,
};
pub use mock_catalog::{
    MockGalaxy, angular_power_spectrum_cl, apparent_magnitude, build_mock_catalog,
    selection_flux_limit,
};
pub use nfw::{
    DELTA_VIRIALIZED, DensityBin, NfwFitResult, NfwProfile, RHO_CRIT0,
    concentration_bhattacharya2013, concentration_duffy2008, concentration_ludlow2016,
    fit_nfw_concentration, measure_density_profile, r200_from_m200, rho_crit_z,
};
pub use pk_correction::{RnModel, a_grid, correct_pk, correct_pk_with_shot_noise, measure_rn};
pub use pk_rsd::{
    LosAxis, PkMultipoleBin, PkRsdBin, PkRsdParams, compute_pk_multipoles, kaiser_multipole_ratios,
    pk_multipoles, pk_redshift_space,
};
pub use power_spectrum::PkBin;
pub use sps_tables::{SpsGrid, Spsband, sps_luminosity};
pub use subfind::{SubfindParams, SubhaloRecord, find_subhalos, local_density_sph};
pub use velocity_profile::{
    VelocityProfileBin, VelocityProfileParams, sigma_1d, velocity_anisotropy, velocity_profile,
};
pub use xray::{
    XrayBin, bremsstrahlung_emissivity, compute_xray_profile, mass_weighted_temperature,
    spectroscopic_temperature, total_xray_luminosity,
};
