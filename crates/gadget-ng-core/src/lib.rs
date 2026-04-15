pub mod config;
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
    GravitySection, IcKind, InitialConditionsSection, OutputSection, PerformanceSection, RunConfig,
    SimulationSection, SnapshotFormat, SolverKind, TimestepSection,
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
