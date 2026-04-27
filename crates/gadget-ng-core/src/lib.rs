pub mod config;
pub mod cosmology;
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
    AgnSection, BFieldKind, ConductionSection, CoolingKind, CosmologySection, CrSection,
    DecompositionConfig, DustSection, EnrichmentSection, FeedbackSection, G_KPC_MSUN_KMPS,
    GravitySection, IcKind, InitialConditionsSection, InsituAnalysisSection, IntegratorKind,
    IsmSection, MacSoftening, MhdSection, ModifiedGravitySection, MolecularSection,
    NormalizationMode, OpeningCriterion, OutputSection, PerformanceSection, RtSection, RunConfig,
    SfcKind, SidmSection, SimulationSection, SnapshotFormat, SolverKind, SphSection,
    TimestepCriterion, TimestepSection, TransferKind, TurbulenceSection, TwoFluidSection,
    UnitsSection, WindParams,
};
pub use cosmology::{
    CosmologyParams, adaptive_dt_cosmo, cosmo_consistency_error, dark_energy_eos,
    density_contrast_rms, g_code_consistent, gravity_coupling_qksl, growth_factor_d,
    growth_factor_d_ratio, growth_rate_f, hubble_param, minimum_image, neutrino_suppression,
    omega_nu_from_mass, peculiar_vrms, wrap_coord, wrap_position,
};
#[cfg(feature = "simd")]
pub use gravity::RayonDirectGravity;
pub use gravity::{
    DirectGravity, GravitySolver, accelerations_all_particles, pairwise_accel_plummer,
};
#[cfg(feature = "simd")]
pub use gravity_simd::SimdDirectGravity;
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

#[cfg(feature = "simd")]
pub use gravity::parallel_direct;
