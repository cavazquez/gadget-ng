//! Core types for gadget-ng: Vec3, Particle, RunConfig, cosmology, ICs (1LPT/2LPT).

pub mod config;
pub mod cosmology;
pub mod dark_matter;
#[cfg(feature = "gpu")]
pub mod gpu_bridge;
pub mod gravity;
#[cfg(feature = "simd")]
pub mod gravity_simd;
pub mod ic;
pub mod ic_2lpt;
pub mod ic_mhd;
pub mod ic_zeldovich;
pub mod modified_gravity;
pub mod particle;
pub mod transfer_fn;
pub mod vec3;

#[cfg(feature = "gpu")]
pub use gadget_ng_gpu::{GpuDirectGravity, GpuParticlesSoA};
#[cfg(feature = "gpu")]
pub use gpu_bridge::GpuParticlesSoAExt;

pub use config::{
    AgnSection, BFieldKind, ConductionSection, ConfigError, CoolingKind, CosmologySection,
    CrSection, DarkMatterModel, DarkMatterSection, DecompositionConfig, DustSection,
    DustSpeciesModel, EnrichmentSection, FeedbackSection, G_KPC_MSUN_KMPS, GravitySection, IcKind,
    InitialConditionsSection, InsituAnalysisSection, IntegratorKind, IsmSection, MacSoftening,
    MhdSection, ModifiedGravitySection, MolecularSection, NormalizationMode, OpeningCriterion,
    OutputSection, PbhHostKind, PerformanceSection, PopIIISection, RtSection, RunConfig, SfcKind,
    SidmSection, SimulationSection, SnapshotFormat, SolverKind, SphSection, StarFormationModel,
    StellarFeedbackMode, TimestepCriterion, TimestepSection, TransferKind, TurbulenceSection,
    TwoFluidSection, UnitsSection, UvBackgroundModel, WindParams,
};
pub use cosmology::{
    CosmologyParams, NeutrinoHierarchyKind, adaptive_dt_cosmo, cosmo_consistency_error,
    dark_energy_eos, density_contrast_rms, g_code_consistent, gravity_coupling_qksl,
    growth_factor_d, growth_factor_d_ratio, growth_rate_f, hubble_param, minimum_image,
    neutrino_suppression, omega_nu_from_mass, peculiar_vrms, split_m_nu_ev, wrap_coord,
    wrap_position,
};
pub use dark_matter::{
    dark_matter_transfer_suppression, fdm_half_mode_k, fdm_quantum_pressure_cs2,
    fdm_transfer_suppression, wdm_half_mode_k, wdm_transfer_suppression,
};
pub use gadget_ng_gpu_layout::{BH_GPU_NO_PARTICLE, BhFmmGpuNode, BhMonopoleGpuNode};
#[cfg(feature = "rayon")]
pub use gravity::RayonDirectGravity;
#[cfg(all(feature = "rayon", feature = "simd"))]
pub use gravity::RayonDirectGravitySimdTier;
pub use gravity::{
    DirectGravity, GravitySolver, accelerations_all_particles, pairwise_accel_plummer,
};
#[cfg(feature = "simd")]
pub use gravity_simd::{GravSimdTier, SimdDirectGravity, SimdDirectGravityTier};
pub use ic::{IcError, build_particles, build_particles_for_gid_range};
pub use ic_2lpt::{Psi2Variant, zeldovich_2lpt_ics, zeldovich_2lpt_ics_with_variant};
pub use ic_mhd::{
    check_plasma_beta, primordial_bfield_ic, primordial_bfield_ic_3d, uniform_bfield_ic,
};
pub use ic_zeldovich::internals as ic_zeldovich_internals;
pub use ic_zeldovich::{IcMomentumConvention, zeldovich_ics, zeldovich_ics_with_convention};
pub use modified_gravity::{FRParams, apply_modified_gravity, chameleon_field, fifth_force_factor};
pub use particle::{Particle, ParticleType};
pub use transfer_fn::{
    EisensteinHuParams, amplitude_for_sigma8, sigma_from_pk_bins, sigma_sq_unit, tophat_window,
    transfer_eh_nowiggle,
};
pub use vec3::Vec3;

#[cfg(feature = "rayon")]
pub use gravity::parallel_direct;
