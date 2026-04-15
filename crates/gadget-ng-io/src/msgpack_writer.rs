//! Snapshot en formato **MessagePack** usando `rmp-serde`.
//!
//! MessagePack es un formato binario compacto e interoperable con Python/R/Julia.
//! Se usa el mismo layout que el bincode writer:
//! - `meta.json` + `provenance.json` (texto, igual que los demás formatos)
//! - `particles.msgpack` — `Vec<ParticleRecord>` serializado con rmp-serde
//!
//! ## Lectura desde Python
//!
//! ```python
//! import msgpack, json, pathlib
//!
//! snap = pathlib.Path("snapshot_final")
//! meta = json.loads((snap / "meta.json").read_text())
//! records = msgpack.unpackb((snap / "particles.msgpack").read_bytes())
//! ```
use std::fs;
use std::path::Path;

use gadget_ng_core::Particle;

use crate::error::SnapshotError;
use crate::provenance::Provenance;
use crate::reader::{SnapshotData, SnapshotReader};
use crate::snapshot::{build_meta, read_meta, write_sidecar_json, ParticleRecord};
use crate::writer::{SnapshotEnv, SnapshotWriter};

const PARTICLES_FILE: &str = "particles.msgpack";

/// Snapshot binario con MessagePack + los mismos sidecar JSON que JSONL.
#[derive(Debug, Default, Clone, Copy)]
pub struct MsgpackWriter;

impl SnapshotWriter for MsgpackWriter {
    fn write(
        &self,
        out_dir: &Path,
        particles: &[Particle],
        provenance: &Provenance,
        env: &SnapshotEnv,
    ) -> Result<(), SnapshotError> {
        let meta = build_meta(particles.len(), provenance, env);
        write_sidecar_json(out_dir, &meta, provenance)?;
        let records: Vec<ParticleRecord> = particles.iter().map(ParticleRecord::from).collect();
        let bytes =
            rmp_serde::to_vec_named(&records).map_err(|e| SnapshotError::Msgpack(e.to_string()))?;
        fs::write(out_dir.join(PARTICLES_FILE), bytes)?;
        Ok(())
    }
}

/// Lector MessagePack: reconstruye partículas y metadatos desde
/// `particles.msgpack` + `meta.json`.
#[derive(Debug, Default, Clone, Copy)]
pub struct MsgpackReader;

impl SnapshotReader for MsgpackReader {
    fn read(&self, dir: &Path) -> Result<SnapshotData, SnapshotError> {
        let meta = read_meta(dir)?;
        let bytes = fs::read(dir.join(PARTICLES_FILE))?;
        let records: Vec<ParticleRecord> =
            rmp_serde::from_slice(&bytes).map_err(|e| SnapshotError::Msgpack(e.to_string()))?;
        let particles = records.into_iter().map(|r| r.into_particle()).collect();
        Ok(SnapshotData {
            particles,
            time: meta.time,
            redshift: meta.redshift,
            box_size: meta.box_size,
        })
    }
}

#[cfg(all(test, feature = "msgpack"))]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    fn dummy_prov() -> Provenance {
        crate::provenance::Provenance::new("0-test", None, "debug", vec![], vec![], "hash")
    }

    #[test]
    fn msgpack_writer_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let particles = vec![
            Particle::new(0, 1.0, Vec3::new(0.1, 0.2, 0.3), Vec3::zero()),
            Particle::new(1, 2.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
        ];
        MsgpackWriter
            .write(
                dir.path(),
                &particles,
                &dummy_prov(),
                &SnapshotEnv::default(),
            )
            .unwrap();
        assert!(dir.path().join(PARTICLES_FILE).exists());
        assert!(dir.path().join("meta.json").exists());
    }

    #[test]
    fn msgpack_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let particles = vec![
            Particle::new(0, 1.5, Vec3::new(0.1, 0.2, 0.3), Vec3::new(0.4, 0.5, 0.6)),
            Particle::new(1, 2.5, Vec3::new(1.0, 2.0, 3.0), Vec3::new(-1.0, 0.0, 1.0)),
        ];
        let env = SnapshotEnv {
            time: 1.23,
            redshift: 0.7,
            box_size: 8.0,
        };
        MsgpackWriter
            .write(dir.path(), &particles, &dummy_prov(), &env)
            .unwrap();
        let data = MsgpackReader.read(dir.path()).unwrap();
        assert_eq!(data.particles.len(), 2);
        assert_eq!(data.particles[0], particles[0]);
        assert_eq!(data.particles[1], particles[1]);
        assert!((data.time - 1.23).abs() < 1e-15);
        assert!((data.box_size - 8.0).abs() < 1e-15);
        assert!((data.redshift - 0.7).abs() < 1e-15);
    }
}
