use std::fs;
use std::path::Path;

use gadget_ng_core::Particle;

use crate::error::SnapshotError;
use crate::provenance::Provenance;
use crate::snapshot::{build_meta, write_sidecar_json, ParticleRecord};
use crate::writer::{SnapshotEnv, SnapshotWriter};

/// Snapshot binario con `bincode` + mismos `meta.json` / `provenance.json` que JSONL.
#[derive(Debug, Default, Clone, Copy)]
pub struct BincodeWriter;

impl SnapshotWriter for BincodeWriter {
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
        let bytes = bincode::serde::encode_to_vec(&records, bincode::config::standard())
            .map_err(|e| SnapshotError::Bincode(e.to_string()))?;
        fs::write(out_dir.join("particles.bin"), bytes)?;
        Ok(())
    }
}

#[cfg(all(test, feature = "bincode"))]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    #[test]
    fn bincode_roundtrip_records() {
        let dir = tempfile::tempdir().unwrap();
        let particles = vec![
            Particle::new(0, 1.0, Vec3::new(0.1, 0.2, 0.3), Vec3::zero()),
            Particle::new(1, 2.0, Vec3::new(1., 0., 0.), Vec3::new(0., 1., 0.)),
        ];
        let prov =
            crate::provenance::Provenance::new("0-test", None, "debug", vec![], vec![], "hash");
        let env = crate::writer::SnapshotEnv::default();
        BincodeWriter
            .write(dir.path(), &particles, &prov, &env)
            .unwrap();

        let bytes = std::fs::read(dir.path().join("particles.bin")).unwrap();
        let (decoded, _len): (Vec<crate::snapshot::ParticleRecord>, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].clone().into_particle(), particles[0]);
        assert_eq!(decoded[1].clone().into_particle(), particles[1]);
    }
}
