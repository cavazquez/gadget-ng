//! Hierarchical block timestep path (Phase 56).
//!
//! TODO: Extraer el bloque `if cfg.timestep.hierarchical { ... }` de mod.rs aquí.
//! Obstrucción: ~15 variables capturadas por los physics hooks (`context::step_*`).
//! Requiere: refactorizar cada branch para pasar las variables explícitamente.
#![allow(dead_code)]

/// Placeholder — el código real reside en `super::run_stepping()` mod.rs.
pub(crate) const HIERARCHICAL_EXTRACTED: bool = false;
