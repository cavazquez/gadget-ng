#[cfg(feature = "bincode")]
mod bincode_writer;
mod error;
pub mod gadget4_attrs;
#[cfg(feature = "hdf5")]
mod hdf5_writer;
pub mod hdf5_parallel_writer;
#[cfg(feature = "msgpack")]
mod msgpack_writer;
#[cfg(feature = "netcdf")]
mod netcdf_writer;
mod provenance;
mod reader;
mod snapshot;
mod writer;

pub use error::SnapshotError;
pub use gadget4_attrs::{Gadget4Header, KMS_IN_CMS, KPC_IN_CM, MSUN_IN_G};
#[cfg(feature = "hdf5")]
pub use gadget4_attrs::{read_gadget4_header, write_gadget4_header};
pub use provenance::Provenance;
pub use reader::{SnapshotData, SnapshotReader};
pub use snapshot::{write_snapshot, JsonlReader, JsonlWriter, ParticleRecord, SnapshotMeta};
pub use writer::{SnapshotEnv, SnapshotUnits, SnapshotWriter};

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
        #[cfg(feature = "msgpack")]
        SnapshotFormat::Msgpack => Ok(Box::new(msgpack_writer::MsgpackWriter)),
        #[cfg(not(feature = "msgpack"))]
        SnapshotFormat::Msgpack => Err(SnapshotError::UnsupportedFormat(
            "msgpack (recompilar con --features msgpack)".into(),
        )),
        #[cfg(feature = "netcdf")]
        SnapshotFormat::Netcdf => Ok(Box::new(netcdf_writer::NetcdfWriter)),
        #[cfg(not(feature = "netcdf"))]
        SnapshotFormat::Netcdf => Err(SnapshotError::UnsupportedFormat(
            "netcdf (recompilar con --features netcdf)".into(),
        )),
    }
}

/// Selecciona el lector según el formato de configuración.
pub fn reader_for(
    fmt: SnapshotFormat,
) -> Result<Box<dyn SnapshotReader + Send + Sync>, SnapshotError> {
    match fmt {
        SnapshotFormat::Jsonl => Ok(Box::new(JsonlReader)),
        #[cfg(feature = "bincode")]
        SnapshotFormat::Bincode => Ok(Box::new(bincode_writer::BincodeReader)),
        #[cfg(not(feature = "bincode"))]
        SnapshotFormat::Bincode => Err(SnapshotError::UnsupportedFormat(
            "bincode (recompilar con --features bincode)".into(),
        )),
        #[cfg(feature = "hdf5")]
        SnapshotFormat::Hdf5 => Ok(Box::new(hdf5_writer::Hdf5Reader)),
        #[cfg(not(feature = "hdf5"))]
        SnapshotFormat::Hdf5 => Err(SnapshotError::UnsupportedFormat(
            "hdf5 (recompilar con --features hdf5)".into(),
        )),
        #[cfg(feature = "msgpack")]
        SnapshotFormat::Msgpack => Ok(Box::new(msgpack_writer::MsgpackReader)),
        #[cfg(not(feature = "msgpack"))]
        SnapshotFormat::Msgpack => Err(SnapshotError::UnsupportedFormat(
            "msgpack (recompilar con --features msgpack)".into(),
        )),
        #[cfg(feature = "netcdf")]
        SnapshotFormat::Netcdf => Ok(Box::new(netcdf_writer::NetcdfReader)),
        #[cfg(not(feature = "netcdf"))]
        SnapshotFormat::Netcdf => Err(SnapshotError::UnsupportedFormat(
            "netcdf (recompilar con --features netcdf)".into(),
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

/// Lee un snapshot desde el directorio indicado, usando el formato de configuración.
pub fn read_snapshot_formatted(
    fmt: SnapshotFormat,
    dir: &Path,
) -> Result<SnapshotData, SnapshotError> {
    reader_for(fmt)?.read(dir)
}
