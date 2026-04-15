pub mod config;
pub mod gravity;
pub mod ic;
pub mod particle;
pub mod vec3;

pub use config::{
    GravitySection, IcKind, InitialConditionsSection, OutputSection, PerformanceSection, RunConfig,
    SimulationSection, SnapshotFormat, SolverKind,
};
#[cfg(feature = "simd")]
pub use gravity::RayonDirectGravity;
pub use gravity::{
    accelerations_all_particles, pairwise_accel_plummer, DirectGravity, GravitySolver,
};
pub use ic::{build_particles, build_particles_for_gid_range, IcError};
pub use particle::Particle;
pub use vec3::Vec3;

#[cfg(feature = "simd")]
pub use gravity::parallel_direct;
