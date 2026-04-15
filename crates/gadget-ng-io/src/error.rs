use thiserror::Error;

#[derive(Debug, Error)]
pub enum SnapshotError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialización JSON: {0}")]
    SerdeJson(#[from] serde_json::Error),
    #[error("formato snapshot no disponible en esta build: {0}")]
    UnsupportedFormat(String),
    #[error("bincode: {0}")]
    Bincode(String),
    #[cfg(feature = "hdf5")]
    #[error("HDF5: {0}")]
    Hdf5(#[from] hdf5::Error),
}
