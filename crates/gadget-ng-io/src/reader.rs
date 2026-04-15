//! Contrato común para lectura de snapshots (JSONL, bincode, HDF5).
use gadget_ng_core::Particle;
use std::path::Path;

use crate::error::SnapshotError;

/// Datos devueltos al leer un snapshot: partículas y metadatos de simulación.
#[derive(Debug, Clone)]
pub struct SnapshotData {
    pub particles: Vec<Particle>,
    pub time: f64,
    pub redshift: f64,
    pub box_size: f64,
}

pub trait SnapshotReader: Send + Sync {
    fn read(&self, dir: &Path) -> Result<SnapshotData, SnapshotError>;
}
