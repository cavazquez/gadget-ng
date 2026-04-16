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
pub mod fof;
pub mod power_spectrum;

pub use catalog::{
    analyse, read_halo_catalog, read_power_spectrum, write_halo_catalog, write_power_spectrum,
    AnalysisParams, AnalysisResult,
};
pub use fof::FofHalo;
pub use power_spectrum::PkBin;
