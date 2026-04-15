//! Contrato común para escritura de snapshots (JSONL, bincode, HDF5).
use gadget_ng_core::Particle;
use std::path::Path;

use crate::error::SnapshotError;
use crate::provenance::Provenance;

/// Metadatos cosmográficos / de corrida pasados al escribir (p. ej. para cabecera HDF5).
#[derive(Debug, Clone)]
pub struct SnapshotEnv {
    pub time: f64,
    pub redshift: f64,
    pub box_size: f64,
}

impl Default for SnapshotEnv {
    fn default() -> Self {
        Self {
            time: 0.0,
            redshift: 0.0,
            box_size: 1.0,
        }
    }
}

pub trait SnapshotWriter: Send + Sync {
    fn write(
        &self,
        out_dir: &Path,
        particles: &[Particle],
        provenance: &Provenance,
        env: &SnapshotEnv,
    ) -> Result<(), SnapshotError>;
}
