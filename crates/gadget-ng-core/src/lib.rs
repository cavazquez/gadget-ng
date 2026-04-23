pub mod config;
pub mod cosmology;
#[cfg(feature = "gpu")]
pub mod gpu_bridge;
pub mod gravity;
#[cfg(feature = "simd")]
pub mod gravity_simd;
pub mod ic;
pub mod ic_2lpt;
pub mod ic_zeldovich;
pub mod particle;
pub mod transfer_fn;
pub mod vec3;

#[cfg(feature = "gpu")]
pub use gadget_ng_gpu::{GpuDirectGravity, GpuParticlesSoA};
#[cfg(feature = "gpu")]
pub use gpu_bridge::GpuParticlesSoAExt;

pub use config::{
    CosmologySection, CoolingKind, DecompositionConfig, GravitySection, IcKind,
    InitialConditionsSection, InsituAnalysisSection, IntegratorKind, MacSoftening,
    NormalizationMode, OpeningCriterion, OutputSection, PerformanceSection, RunConfig, SfcKind,
    SimulationSection, SnapshotFormat, SolverKind, SphSection, TimestepCriterion, TimestepSection,
    TransferKind, UnitsSection, G_KPC_MSUN_KMPS,
};
pub use cosmology::{
    adaptive_dt_cosmo, cosmo_consistency_error, density_contrast_rms, g_code_consistent,
    gravity_coupling_qksl, growth_factor_d, growth_factor_d_ratio, growth_rate_f, hubble_param,
    minimum_image, peculiar_vrms, wrap_coord, wrap_position, CosmologyParams,
};
#[cfg(feature = "simd")]
pub use gravity::RayonDirectGravity;
pub use gravity::{
    accelerations_all_particles, pairwise_accel_plummer, DirectGravity, GravitySolver,
};
#[cfg(feature = "simd")]
pub use gravity_simd::SimdDirectGravity;
pub use ic::{build_particles, build_particles_for_gid_range, IcError};
pub use ic_2lpt::{zeldovich_2lpt_ics, zeldovich_2lpt_ics_with_variant, Psi2Variant};
pub use ic_zeldovich::internals as ic_zeldovich_internals;
pub use ic_zeldovich::{zeldovich_ics, zeldovich_ics_with_convention, IcMomentumConvention};
pub use particle::{Particle, ParticleType};
pub use transfer_fn::{
    amplitude_for_sigma8, sigma_from_pk_bins, sigma_sq_unit, tophat_window, transfer_eh_nowiggle,
    EisensteinHuParams,
};
pub use vec3::Vec3;

#[cfg(feature = "simd")]
pub use gravity::parallel_direct;
