#[cfg(feature = "bincode")]
mod bincode_writer;
mod error;
#[cfg(feature = "hdf5")]
mod hdf5_writer;
mod provenance;
mod snapshot;
mod writer;

pub use error::SnapshotError;
pub use provenance::Provenance;
pub use snapshot::{write_snapshot, JsonlWriter, ParticleRecord};
pub use writer::{SnapshotEnv, SnapshotWriter};

use gadget_ng_core::{Particle, SnapshotFormat};
use std::path::Path;

/// Selecciona el escritor según el formato de configuración.
pub fn writer_for(
    fmt: SnapshotFormat,
) -> Result<Box<dyn SnapshotWriter + Send + Sync>, SnapshotError> {
    match fmt {
        SnapshotFormat::Jsonl => Ok(Box::new(JsonlWriter)),
        #[cfg(feature = "bincode")]
        SnapshotFormat::Bincode => Ok(Box::new(bincode_writer::BincodeWriter)),
        #[cfg(not(feature = "bincode"))]
        SnapshotFormat::Bincode => Err(SnapshotError::UnsupportedFormat(
            "bincode (recompilar con --features bincode)".into(),
        )),
        #[cfg(feature = "hdf5")]
        SnapshotFormat::Hdf5 => Ok(Box::new(hdf5_writer::Hdf5Writer)),
        #[cfg(not(feature = "hdf5"))]
        SnapshotFormat::Hdf5 => Err(SnapshotError::UnsupportedFormat(
            "hdf5 (recompilar con --features hdf5)".into(),
        )),
    }
}

/// Escribe un snapshot usando el formato indicado y el entorno (tiempo, redshift, caja).
pub fn write_snapshot_formatted(
    fmt: SnapshotFormat,
    out_dir: &Path,
    particles: &[Particle],
    provenance: &Provenance,
    env: &SnapshotEnv,
) -> Result<(), SnapshotError> {
    writer_for(fmt)?.write(out_dir, particles, provenance, env)
}
