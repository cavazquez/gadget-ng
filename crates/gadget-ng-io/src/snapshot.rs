use crate::error::SnapshotError;
use crate::provenance::Provenance;
use crate::writer::{SnapshotEnv, SnapshotWriter};
use gadget_ng_core::Particle;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ParticleRecord {
    pub global_id: usize,
    pub mass: f64,
    pub px: f64,
    pub py: f64,
    pub pz: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
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

impl ParticleRecord {
    pub fn into_particle(self) -> Particle {
        use gadget_ng_core::Vec3;
        Particle::new(
            self.global_id,
            self.mass,
            Vec3::new(self.px, self.py, self.pz),
            Vec3::new(self.vx, self.vy, self.vz),
        )
    }
}

#[derive(Serialize, Clone)]
pub(crate) struct SnapshotMeta {
    pub schema_version: u32,
    pub provenance: Provenance,
    pub particle_count: usize,
    pub time: f64,
    pub redshift: f64,
    pub box_size: f64,
}

pub(crate) fn build_meta(
    particle_count: usize,
    provenance: &Provenance,
    env: &SnapshotEnv,
) -> SnapshotMeta {
    SnapshotMeta {
        schema_version: 1,
        provenance: provenance.clone(),
        particle_count,
        time: env.time,
        redshift: env.redshift,
        box_size: env.box_size,
    }
}

pub(crate) fn write_sidecar_json(
    out_dir: &Path,
    meta: &SnapshotMeta,
    provenance: &Provenance,
) -> Result<(), SnapshotError> {
    fs::create_dir_all(out_dir)?;
    fs::write(
        out_dir.join("meta.json"),
        serde_json::to_string_pretty(meta)?,
    )?;
    fs::write(
        out_dir.join("provenance.json"),
        serde_json::to_string_pretty(provenance)?,
    )?;
    Ok(())
}

/// Escritor JSONL (comportamiento original del MVP).
#[derive(Debug, Default, Clone, Copy)]
pub struct JsonlWriter;

impl SnapshotWriter for JsonlWriter {
    fn write(
        &self,
        out_dir: &Path,
        particles: &[Particle],
        provenance: &Provenance,
        env: &SnapshotEnv,
    ) -> Result<(), SnapshotError> {
        let meta = build_meta(particles.len(), provenance, env);
        write_sidecar_json(out_dir, &meta, provenance)?;
        let mut lines = String::new();
        for p in particles {
            let line = serde_json::to_string(&ParticleRecord::from(p))?;
            lines.push_str(&line);
            lines.push('\n');
        }
        fs::write(out_dir.join("particles.jsonl"), lines)?;
        Ok(())
    }
}

/// Escribe `meta.json`, `provenance.json` y `particles.jsonl` (API legada).
pub fn write_snapshot(
    out_dir: &Path,
    particles: &[Particle],
    provenance: &Provenance,
) -> Result<(), SnapshotError> {
    JsonlWriter.write(out_dir, particles, provenance, &SnapshotEnv::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;
    use std::fs;

    fn dummy_provenance() -> Provenance {
        Provenance::new(
            "0-test",
            None,
            "debug",
            vec![],
            vec!["gadget-ng".into()],
            "abc123",
        )
    }

    #[test]
    fn jsonl_roundtrip_particles_match() {
        let dir = tempfile::tempdir().unwrap();
        let particles = vec![
            Particle::new(0, 1.0, Vec3::new(0.1, 0.2, 0.3), Vec3::new(0., 0., 0.)),
            Particle::new(1, 2.0, Vec3::new(1., 0., 0.), Vec3::new(0., 1., 0.)),
        ];
        let prov = dummy_provenance();
        let env = SnapshotEnv {
            time: 0.05,
            redshift: 0.0,
            box_size: 1.0,
        };
        JsonlWriter
            .write(dir.path(), &particles, &prov, &env)
            .unwrap();

        let meta: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.path().join("meta.json")).unwrap())
                .unwrap();
        assert_eq!(meta["particle_count"], 2);
        assert_eq!(meta["time"], 0.05);
        assert_eq!(meta["box_size"], 1.0);

        let lines: Vec<String> = fs::read_to_string(dir.path().join("particles.jsonl"))
            .unwrap()
            .lines()
            .map(String::from)
            .collect();
        assert_eq!(lines.len(), 2);
        let p0: ParticleRecord = serde_json::from_str(&lines[0]).unwrap();
        let p1: ParticleRecord = serde_json::from_str(&lines[1]).unwrap();
        assert_eq!(p0.into_particle(), particles[0]);
        assert_eq!(p1.into_particle(), particles[1]);
    }
}
