pub mod config;
pub mod cosmology;
#[cfg(feature = "gpu")]
pub mod gpu_bridge;
pub mod gravity;
#[cfg(feature = "simd")]
pub mod gravity_simd;
pub mod ic;
pub mod particle;
pub mod vec3;

#[cfg(feature = "gpu")]
pub use gadget_ng_gpu::{GpuDirectGravity, GpuParticlesSoA};
#[cfg(feature = "gpu")]
pub use gpu_bridge::GpuParticlesSoAExt;

pub use config::{
    CosmologySection, GravitySection, IcKind, InitialConditionsSection, IntegratorKind,
    MacSoftening, OpeningCriterion, OutputSection, PerformanceSection, RunConfig,
    SfcKind, SimulationSection, SnapshotFormat, SolverKind, TimestepCriterion, TimestepSection,
    UnitsSection, G_KPC_MSUN_KMPS,
};
pub use cosmology::{
    density_contrast_rms, hubble_param, minimum_image, peculiar_vrms, wrap_coord, wrap_position,
    CosmologyParams,
};
#[cfg(feature = "simd")]
pub use gravity::RayonDirectGravity;
pub use gravity::{
    accelerations_all_particles, pairwise_accel_plummer, DirectGravity, GravitySolver,
};
#[cfg(feature = "simd")]
pub use gravity_simd::SimdDirectGravity;
pub use ic::{build_particles, build_particles_for_gid_range, IcError};
pub use particle::Particle;
pub use vec3::Vec3;

#[cfg(feature = "simd")]
pub use gravity::parallel_direct;
