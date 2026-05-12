//! Tipos de configuración por dominio (deserialización TOML).
//!
//! Cada submódulo agrupa secciones relacionadas; aquí se reexportan para rutas
//! estables (`crate::config::SimulationSection`, etc.).

mod analysis;
mod cosmology_units;
mod extended;
mod gravity;
mod mhd;
mod output_perf;
mod rt_reion;
mod simulation_ic;
mod sph;
mod timestep;

pub use analysis::{DecompositionConfig, InsituAnalysisSection};
pub use cosmology_units::{CosmologySection, G_KPC_MSUN_KMPS, UnitsSection};
pub use extended::{
    DarkMatterModel, DarkMatterSection, ModifiedGravitySection, SidmSection, TurbulenceSection,
    TwoFluidSection,
};
pub use gravity::{GravitySection, MacSoftening, OpeningCriterion, SolverKind};
pub use mhd::{BFieldKind, MhdSection};
pub use output_perf::{OutputSection, PerformanceSection, SfcKind, SnapshotFormat};
pub use rt_reion::{ReionizationSection, RtSection};
pub use simulation_ic::{
    IcKind, InitialConditionsSection, IntegratorKind, NormalizationMode, SimulationSection,
    TransferKind,
};
pub use sph::{
    AgnSection, ConductionSection, CoolingKind, CrSection, DustSection, EnrichmentSection,
    FeedbackSection, IsmSection, MolecularSection, PopIIISection, SphSection, StarFormationModel,
    StellarFeedbackMode, UvBackgroundModel, WindParams,
};
pub use timestep::{TimestepCriterion, TimestepSection};
