//! Contrato común para escritura de snapshots (JSONL, bincode, HDF5).
use gadget_ng_core::Particle;
use std::path::Path;

use crate::error::SnapshotError;
use crate::provenance::Provenance;

/// Metadatos de unidades físicas para incluir en `meta.json`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SnapshotUnits {
    /// 1 unidad interna de longitud = `length_in_kpc` kpc.
    pub length_in_kpc: f64,
    /// 1 unidad interna de masa = `mass_in_msun` Msun.
    pub mass_in_msun: f64,
    /// 1 unidad interna de velocidad = `velocity_in_km_s` km/s.
    pub velocity_in_km_s: f64,
    /// Unidad de tiempo interna en Gyr (= 0.97779 × length_in_kpc / velocity_in_km_s).
    pub time_in_gyr: f64,
    /// G en unidades internas.
    pub g_internal: f64,
}

/// Metadatos cosmográficos / de corrida pasados al escribir (p. ej. para cabecera HDF5).
#[derive(Debug, Clone)]
pub struct SnapshotEnv {
    pub time: f64,
    pub redshift: f64,
    pub box_size: f64,
    /// `Some(...)` si la simulación usa unidades físicas (`[units] enabled = true`).
    pub units: Option<SnapshotUnits>,
    /// Ω_m (materia). 0.0 si no hay cosmología activa.
    pub omega_m: f64,
    /// Ω_Λ (constante cosmológica). 0.0 si no hay cosmología activa.
    pub omega_lambda: f64,
    /// H₀/100 (adimensional). 1.0 si no hay cosmología activa.
    pub h_dimless: f64,
}

impl Default for SnapshotEnv {
    fn default() -> Self {
        Self {
            time: 0.0,
            redshift: 0.0,
            box_size: 1.0,
            units: None,
            omega_m: 0.0,
            omega_lambda: 0.0,
            h_dimless: 1.0,
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
