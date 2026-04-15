use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize)]
pub struct Provenance {
    pub crate_version: String,
    pub git_commit: Option<String>,
    pub build_profile: String,
    pub enabled_features: Vec<String>,
    pub command_line: Vec<String>,
    pub config_hash: String,
}

impl Provenance {
    pub fn new(
        crate_version: impl Into<String>,
        git_commit: Option<String>,
        build_profile: impl Into<String>,
        enabled_features: Vec<String>,
        command_line: Vec<String>,
        config_hash: impl Into<String>,
    ) -> Self {
        Self {
            crate_version: crate_version.into(),
            git_commit,
            build_profile: build_profile.into(),
            enabled_features,
            command_line,
            config_hash: config_hash.into(),
        }
    }

    pub fn json_sha256(&self) -> String {
        let s = serde_json::to_string(self).expect("serialize provenance");
        let h = Sha256::digest(s.as_bytes());
        hex::encode(h)
    }
}
