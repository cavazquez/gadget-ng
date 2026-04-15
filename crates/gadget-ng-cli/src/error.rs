use figment::Error as FigmentError;
use gadget_ng_core::IcError;
use gadget_ng_io::SnapshotError;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CliError {
    #[error("configuración: {0}")]
    Config(Box<FigmentError>),
    #[error("I/O {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("condiciones iniciales: {0}")]
    Ic(#[from] IcError),
    #[error("snapshot: {0}")]
    Snapshot(#[from] SnapshotError),
    #[error("TOML inválido: {0}")]
    TomlSer(#[from] toml::ser::Error),
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
}

impl From<FigmentError> for CliError {
    fn from(value: FigmentError) -> Self {
        Self::Config(Box::new(value))
    }
}

impl CliError {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
