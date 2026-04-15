mod provenance;
mod snapshot;

pub use provenance::Provenance;
pub use snapshot::{write_snapshot, SnapshotError};

#[cfg(feature = "netcdf")]
pub mod netcdf_stub {
    //! Contrato NetCDF (placeholder): escritura no implementada en el MVP.

    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum NetCdfError {
        #[error("salida NetCDF aún no implementada; usar snapshot JSONL")]
        Unimplemented,
    }

    pub fn write_netcdf_placeholder() -> Result<(), NetCdfError> {
        Err(NetCdfError::Unimplemented)
    }
}
