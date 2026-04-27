//! Phase 105 — JSONL con campos SPH
//!
//! Verifica que `ParticleRecord` persiste `internal_energy`, `smoothing_length`
//! y `ptype` en snapshots JSONL, y que la lectura retrocompatible funciona con
//! archivos generados antes del Phase 105.

use gadget_ng_core::{Particle, ParticleType, Vec3};
use gadget_ng_io::{
    JsonlReader, JsonlWriter, ParticleRecord, Provenance, SnapshotEnv, SnapshotReader,
    SnapshotWriter,
};
use std::fs;
use tempfile::tempdir;

fn provenance() -> Provenance {
    Provenance::new(
        "phase105",
        None,
        "debug",
        vec![],
        vec!["gadget-ng".into()],
        "phase105",
    )
}

fn env() -> SnapshotEnv {
    SnapshotEnv {
        time: 1.0,
        redshift: 0.0,
        box_size: 10.0,
        units: None,
        ..Default::default()
    }
}

fn gas(id: usize, u: f64, h: f64) -> Particle {
    Particle::new_gas(
        id,
        1.0,
        Vec3::new(id as f64 * 0.5, 0.0, 0.0),
        Vec3::zero(),
        u,
        h,
    )
}

fn dm(id: usize) -> Particle {
    Particle::new(id, 2.0, Vec3::new(id as f64 * 0.3, 0.1, 0.2), Vec3::zero())
}

/// Los campos SPH de una partícula gas se preservan en roundtrip JSONL.
#[test]
fn gas_sph_fields_survive_roundtrip() {
    let dir = tempdir().unwrap();
    let particles = vec![gas(0, 4.2, 0.07), gas(1, 9.1, 0.13)];
    JsonlWriter
        .write(dir.path(), &particles, &provenance(), &env())
        .unwrap();
    let data = JsonlReader.read(dir.path()).unwrap();
    assert_eq!(data.particles.len(), 2);
    for (orig, back) in particles.iter().zip(data.particles.iter()) {
        assert_eq!(back.ptype, ParticleType::Gas, "ptype debe ser Gas");
        assert!(
            (back.internal_energy - orig.internal_energy).abs() < 1e-14,
            "u: expected {}, got {}",
            orig.internal_energy,
            back.internal_energy
        );
        assert!(
            (back.smoothing_length - orig.smoothing_length).abs() < 1e-14,
            "h: expected {}, got {}",
            orig.smoothing_length,
            back.smoothing_length
        );
    }
}

/// DM particles tienen ptype=DarkMatter y SPH fields = 0 tras roundtrip.
#[test]
fn dm_particle_ptype_preserved() {
    let dir = tempdir().unwrap();
    let particles = vec![dm(0), dm(1)];
    JsonlWriter
        .write(dir.path(), &particles, &provenance(), &env())
        .unwrap();
    let data = JsonlReader.read(dir.path()).unwrap();
    for p in &data.particles {
        assert_eq!(p.ptype, ParticleType::DarkMatter);
        assert_eq!(p.internal_energy, 0.0);
        assert_eq!(p.smoothing_length, 0.0);
    }
}

/// Snapshot mixto (DM + gas) roundtrip.
#[test]
fn mixed_snapshot_roundtrip() {
    let dir = tempdir().unwrap();
    let particles = vec![dm(0), gas(1, 3.3, 0.09), dm(2), gas(3, 6.6, 0.04)];
    JsonlWriter
        .write(dir.path(), &particles, &provenance(), &env())
        .unwrap();
    let data = JsonlReader.read(dir.path()).unwrap();
    assert_eq!(data.particles.len(), 4);
    assert_eq!(data.particles[0].ptype, ParticleType::DarkMatter);
    assert_eq!(data.particles[1].ptype, ParticleType::Gas);
    assert!((data.particles[1].internal_energy - 3.3).abs() < 1e-14);
    assert_eq!(data.particles[2].ptype, ParticleType::DarkMatter);
    assert_eq!(data.particles[3].ptype, ParticleType::Gas);
    assert!((data.particles[3].smoothing_length - 0.04).abs() < 1e-14);
}

/// Compatibilidad retroactiva: JSONL viejo (sin campos SPH) se lee con defaults.
#[test]
fn legacy_jsonl_backward_compat() {
    let dir = tempdir().unwrap();
    // Escribir JSONL viejo manualmente (sin campos SPH).
    // El meta.json debe tener un Provenance válido (crate_version, etc.).
    fs::create_dir_all(dir.path()).unwrap();
    let prov_json = r#"{"crate_version":"0","git_commit":null,"build_profile":"debug","enabled_features":[],"command_line":[],"config_hash":"abc"}"#;
    let meta_json = format!(
        r#"{{"schema_version":1,"provenance":{},"particle_count":2,"time":0.5,"redshift":1.0,"box_size":5.0}}"#,
        prov_json
    );
    fs::write(dir.path().join("meta.json"), &meta_json).unwrap();
    fs::write(dir.path().join("provenance.json"), prov_json).unwrap();
    // JSONL sin campos SPH (formato pre-Phase-105)
    let legacy_lines = concat!(
        r#"{"global_id":0,"mass":1.0,"px":0.1,"py":0.2,"pz":0.3,"vx":0.0,"vy":0.0,"vz":0.0}"#,
        "\n",
        r#"{"global_id":1,"mass":2.0,"px":0.4,"py":0.5,"pz":0.6,"vx":0.1,"vy":0.1,"vz":0.1}"#,
        "\n",
    );
    fs::write(dir.path().join("particles.jsonl"), legacy_lines).unwrap();

    let data = JsonlReader.read(dir.path()).unwrap();
    assert_eq!(data.particles.len(), 2);
    for p in &data.particles {
        assert_eq!(
            p.ptype,
            ParticleType::DarkMatter,
            "default ptype debe ser DarkMatter"
        );
        assert_eq!(p.internal_energy, 0.0);
        assert_eq!(p.smoothing_length, 0.0);
    }
}

/// ParticleRecord::from serializa campos SPH correctamente.
#[test]
fn particle_record_serializes_sph_fields() {
    let g = gas(0, 2.5, 0.06);
    let rec = ParticleRecord::from(&g);
    let json = serde_json::to_string(&rec).unwrap();
    assert!(
        json.contains(r#""internal_energy":2.5"#),
        "debe incluir internal_energy: {json}"
    );
    assert!(
        json.contains(r#""smoothing_length":0.06"#),
        "debe incluir smoothing_length: {json}"
    );
    assert!(
        json.contains(r#""ptype":"Gas""#),
        "debe incluir ptype Gas: {json}"
    );
}

/// ParticleRecord::into_particle restaura los campos SPH.
#[test]
fn particle_record_into_particle_restores_sph() {
    let rec = ParticleRecord {
        global_id: 5,
        mass: 1.5,
        px: 1.0,
        py: 2.0,
        pz: 3.0,
        vx: 0.1,
        vy: 0.2,
        vz: 0.3,
        internal_energy: 8.8,
        smoothing_length: 0.15,
        ptype: ParticleType::Gas,
    };
    let p = rec.into_particle();
    assert_eq!(p.global_id, 5);
    assert_eq!(p.ptype, ParticleType::Gas);
    assert!((p.internal_energy - 8.8).abs() < 1e-14);
    assert!((p.smoothing_length - 0.15).abs() < 1e-14);
}
