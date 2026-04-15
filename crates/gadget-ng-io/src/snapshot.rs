use crate::provenance::Provenance;
use gadget_ng_core::Particle;
use serde::Serialize;
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SnapshotError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialización: {0}")]
    Serde(#[from] serde_json::Error),
}

#[derive(Serialize)]
struct SnapshotMeta {
    schema_version: u32,
    provenance: Provenance,
    particle_count: usize,
}

/// Escribe `provenance.json`, `meta.json` y `particles.jsonl` bajo `out_dir`.
pub fn write_snapshot(
    out_dir: &Path,
    particles: &[Particle],
    provenance: &Provenance,
) -> Result<(), SnapshotError> {
    fs::create_dir_all(out_dir)?;
    let meta = SnapshotMeta {
        schema_version: 1,
        provenance: provenance.clone(),
        particle_count: particles.len(),
    };
    fs::write(
        out_dir.join("meta.json"),
        serde_json::to_string_pretty(&meta)?,
    )?;
    fs::write(
        out_dir.join("provenance.json"),
        serde_json::to_string_pretty(&provenance)?,
    )?;
    let mut lines = String::new();
    for p in particles {
        let line = serde_json::to_string(&ParticleRecord::from(p))?;
        lines.push_str(&line);
        lines.push('\n');
    }
    fs::write(out_dir.join("particles.jsonl"), lines)?;
    Ok(())
}

#[derive(Serialize)]
struct ParticleRecord {
    global_id: usize,
    mass: f64,
    px: f64,
    py: f64,
    pz: f64,
    vx: f64,
    vy: f64,
    vz: f64,
}

impl From<&Particle> for ParticleRecord {
    fn from(p: &Particle) -> Self {
        Self {
            global_id: p.global_id,
            mass: p.mass,
            px: p.position.x,
            py: p.position.y,
            pz: p.position.z,
            vx: p.velocity.x,
            vy: p.velocity.y,
            vz: p.velocity.z,
        }
    }
}
