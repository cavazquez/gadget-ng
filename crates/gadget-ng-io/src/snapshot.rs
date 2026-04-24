use crate::error::SnapshotError;
use crate::provenance::Provenance;
use crate::reader::{SnapshotData, SnapshotReader};
use crate::writer::{SnapshotEnv, SnapshotUnits, SnapshotWriter};
use gadget_ng_core::{Particle, ParticleType};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Representación serializable de una partícula en snapshots JSONL.
///
/// Los campos SPH (`internal_energy`, `smoothing_length`, `ptype`) usan
/// `#[serde(default)]` para mantener compatibilidad con snapshots anteriores
/// al Phase 105 que no los incluían.
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
    /// Energía interna específica u [unidades internas]. `0.0` para DM (Phase 105).
    #[serde(default)]
    pub internal_energy: f64,
    /// Radio de suavizado SPH h_sml [unidades internas]. `0.0` para DM (Phase 105).
    #[serde(default)]
    pub smoothing_length: f64,
    /// Tipo de partícula. Default `DarkMatter` para compatibilidad retroactiva (Phase 105).
    #[serde(default)]
    pub ptype: ParticleType,
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
            internal_energy: p.internal_energy,
            smoothing_length: p.smoothing_length,
            ptype: p.ptype,
        }
    }
}

impl ParticleRecord {
    pub fn into_particle(self) -> Particle {
        use gadget_ng_core::Vec3;
        let mut p = Particle::new(
            self.global_id,
            self.mass,
            Vec3::new(self.px, self.py, self.pz),
            Vec3::new(self.vx, self.vy, self.vz),
        );
        p.internal_energy = self.internal_energy;
        p.smoothing_length = self.smoothing_length;
        p.ptype = self.ptype;
        p
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SnapshotMeta {
    pub schema_version: u32,
    pub provenance: Provenance,
    pub particle_count: usize,
    pub time: f64,
    pub redshift: f64,
    pub box_size: f64,
    /// Unidades físicas usadas en la simulación (ausente si `units.enabled = false`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub units: Option<SnapshotUnits>,
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
        units: env.units.clone(),
    }
}

pub(crate) fn read_meta(dir: &Path) -> Result<SnapshotMeta, SnapshotError> {
    let s = fs::read_to_string(dir.join("meta.json"))?;
    Ok(serde_json::from_str(&s)?)
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

/// Lector JSONL: reconstruye partículas y metadatos a partir de un directorio de snapshot.
#[derive(Debug, Default, Clone, Copy)]
pub struct JsonlReader;

impl SnapshotReader for JsonlReader {
    fn read(&self, dir: &Path) -> Result<SnapshotData, SnapshotError> {
        let meta = read_meta(dir)?;
        let file = fs::File::open(dir.join("particles.jsonl"))?;
        let particles = BufReader::new(file)
            .lines()
            .map(|l| {
                let line = l?;
                let rec: ParticleRecord = serde_json::from_str(&line)?;
                Ok(rec.into_particle())
            })
            .collect::<Result<Vec<_>, SnapshotError>>()?;
        Ok(SnapshotData {
            particles,
            time: meta.time,
            redshift: meta.redshift,
            box_size: meta.box_size,
        })
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

    fn gas_particle(id: usize, u: f64, h: f64) -> Particle {
        Particle::new_gas(
            id,
            1.0,
            Vec3::new(id as f64 * 0.1, 0.0, 0.0),
            Vec3::zero(),
            u,
            h,
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
            units: None,
            ..Default::default()
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

    #[test]
    fn jsonl_reader_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let particles = vec![
            Particle::new(0, 1.5, Vec3::new(0.1, 0.2, 0.3), Vec3::new(0.4, 0.5, 0.6)),
            Particle::new(1, 2.5, Vec3::new(1.0, 2.0, 3.0), Vec3::new(-1., 0., 1.)),
        ];
        let prov = dummy_provenance();
        let env = SnapshotEnv {
            time: 0.42,
            redshift: 1.0,
            box_size: 5.0,
            units: None,
            ..Default::default()
        };
        JsonlWriter
            .write(dir.path(), &particles, &prov, &env)
            .unwrap();
        let data = JsonlReader.read(dir.path()).unwrap();
        assert_eq!(data.particles.len(), 2);
        assert_eq!(data.particles[0], particles[0]);
        assert_eq!(data.particles[1], particles[1]);
        assert!((data.time - 0.42).abs() < 1e-15);
        assert!((data.box_size - 5.0).abs() < 1e-15);
        assert!((data.redshift - 1.0).abs() < 1e-15);
    }

    /// Phase 105: gas particles preserve internal_energy and smoothing_length.
    #[test]
    fn gas_particle_sph_fields_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let particles = vec![
            gas_particle(0, 3.5, 0.05),
            gas_particle(1, 7.2, 0.12),
        ];
        let prov = dummy_provenance();
        let env = SnapshotEnv { time: 1.0, redshift: 0.0, box_size: 10.0, units: None, ..Default::default() };
        JsonlWriter.write(dir.path(), &particles, &prov, &env).unwrap();
        let data = JsonlReader.read(dir.path()).unwrap();
        assert_eq!(data.particles.len(), 2);
        for (orig, restored) in particles.iter().zip(data.particles.iter()) {
            assert_eq!(restored.ptype, ParticleType::Gas);
            assert!((restored.internal_energy - orig.internal_energy).abs() < 1e-14);
            assert!((restored.smoothing_length - orig.smoothing_length).abs() < 1e-14);
        }
    }

    /// Phase 105: mixed DM + gas snapshot roundtrip.
    #[test]
    fn mixed_dm_gas_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let dm = Particle::new(0, 2.0, Vec3::new(0.5, 0.5, 0.5), Vec3::zero());
        let gas = gas_particle(1, 5.0, 0.08);
        let particles = vec![dm.clone(), gas.clone()];
        let prov = dummy_provenance();
        let env = SnapshotEnv { time: 0.5, redshift: 0.5, box_size: 1.0, units: None, ..Default::default() };
        JsonlWriter.write(dir.path(), &particles, &prov, &env).unwrap();
        let data = JsonlReader.read(dir.path()).unwrap();
        assert_eq!(data.particles[0].ptype, ParticleType::DarkMatter);
        assert_eq!(data.particles[0].internal_energy, 0.0);
        assert_eq!(data.particles[1].ptype, ParticleType::Gas);
        assert!((data.particles[1].internal_energy - 5.0).abs() < 1e-14);
        assert!((data.particles[1].smoothing_length - 0.08).abs() < 1e-14);
    }

    /// Phase 105: backward compatibility — JSONL without SPH fields deserializes with defaults.
    #[test]
    fn backward_compat_no_sph_fields() {
        let legacy_line = r#"{"global_id":0,"mass":1.0,"px":0.1,"py":0.2,"pz":0.3,"vx":0.0,"vy":0.0,"vz":0.0}"#;
        let rec: ParticleRecord = serde_json::from_str(legacy_line).unwrap();
        assert_eq!(rec.internal_energy, 0.0);
        assert_eq!(rec.smoothing_length, 0.0);
        assert_eq!(rec.ptype, ParticleType::DarkMatter);
        let p = rec.into_particle();
        assert_eq!(p.ptype, ParticleType::DarkMatter);
        assert_eq!(p.internal_energy, 0.0);
    }

    /// Phase 105: ParticleRecord serializes ptype as enum variant name.
    #[test]
    fn ptype_serialized_as_enum_variant() {
        let gas = gas_particle(0, 1.0, 0.1);
        let rec = ParticleRecord::from(&gas);
        let json = serde_json::to_string(&rec).unwrap();
        assert!(json.contains("Gas"), "ptype should serialize as 'Gas': {json}");
        assert!(json.contains("internal_energy"), "should contain internal_energy: {json}");
        assert!(json.contains("smoothing_length"), "should contain smoothing_length: {json}");
    }
}
