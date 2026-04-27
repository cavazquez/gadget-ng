use crate::error::CliError;
use gadget_ng_core::RunConfig;
use sha2::{Digest, Sha256};
use std::path::Path;

use figment::providers::Format;
use figment::{Figment, providers::Env};

pub fn load_run_config(path: &Path) -> Result<RunConfig, CliError> {
    let figment = Figment::new()
        .merge(figment::providers::Toml::file(path))
        .merge(Env::prefixed("GADGET_NG_").split("__"));
    figment.extract().map_err(Into::into)
}

pub fn config_canonical_hash(cfg: &RunConfig) -> Result<String, toml::ser::Error> {
    let s = toml::to_string(cfg)?;
    let h = Sha256::digest(s.as_bytes());
    Ok(hex::encode(h))
}

pub fn print_resolved_config(cfg: &RunConfig) -> Result<(), serde_json::Error> {
    println!("{}", serde_json::to_string_pretty(cfg)?);
    Ok(())
}
