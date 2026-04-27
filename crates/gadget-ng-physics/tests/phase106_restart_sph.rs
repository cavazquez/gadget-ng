//! Phase 106 — Restart con SPH state completo
//!
//! Verifica que el estado SPH (a través de los campos de partícula gracias al
//! Phase 105), el estado AGN (BlackHole vector) y los estados de química EoR
//! (ChemState vector) se persisten correctamente en el directorio de checkpoint
//! y se restauran al reanudar.
//!
//! Nota: este test no ejercita directamente `save_checkpoint`/`load_checkpoint`
//! (funciones privadas de `gadget-ng-cli`) sino que valida la serialización
//! de los tipos involucrados y el formato de los ficheros JSON generados.

use gadget_ng_core::{Particle, ParticleType, Vec3};
use gadget_ng_io::{
    JsonlReader, JsonlWriter, Provenance, SnapshotEnv, SnapshotReader, SnapshotWriter,
};
use gadget_ng_rt::ChemState;
use gadget_ng_sph::BlackHole;
use std::fs;
use tempfile::tempdir;

fn prov() -> Provenance {
    Provenance::new("phase106", None, "debug", vec![], vec![], "hash106")
}

fn env() -> SnapshotEnv {
    SnapshotEnv {
        time: 1.0,
        redshift: 0.5,
        box_size: 10.0,
        units: None,
        ..Default::default()
    }
}

fn gas(id: usize) -> Particle {
    Particle::new_gas(
        id,
        1.0,
        Vec3::new(id as f64, 0.0, 0.0),
        Vec3::zero(),
        3.5 + id as f64,
        0.05,
    )
}

// ── Helpers de serialización ────────────────────────────────────────────────

fn save_agn_bhs(dir: &std::path::Path, bhs: &[BlackHole]) -> std::path::PathBuf {
    let path = dir.join("agn_bhs.json");
    fs::write(&path, serde_json::to_string_pretty(bhs).unwrap()).unwrap();
    path
}

fn load_agn_bhs(dir: &std::path::Path) -> Vec<BlackHole> {
    let s = fs::read_to_string(dir.join("agn_bhs.json")).unwrap();
    serde_json::from_str(&s).unwrap()
}

fn save_chem_states(dir: &std::path::Path, states: &[ChemState]) -> std::path::PathBuf {
    let path = dir.join("chem_states.json");
    fs::write(&path, serde_json::to_string_pretty(states).unwrap()).unwrap();
    path
}

fn load_chem_states(dir: &std::path::Path) -> Vec<ChemState> {
    let s = fs::read_to_string(dir.join("chem_states.json")).unwrap();
    serde_json::from_str(&s).unwrap()
}

// ── Tests ────────────────────────────────────────────────────────────────────

/// BlackHole se serializa y deserializa correctamente (Serde derive Phase 106).
#[test]
fn black_hole_serde_roundtrip() {
    let bh = BlackHole::new(Vec3::new(1.0, 2.0, 3.0), 1e6);
    let json = serde_json::to_string(&bh).unwrap();
    assert!(json.contains("mass"), "debe tener campo mass: {json}");
    let restored: BlackHole = serde_json::from_str(&json).unwrap();
    assert!((restored.mass - 1e6).abs() < 1.0);
    assert!((restored.pos.x - 1.0).abs() < 1e-14);
    assert_eq!(restored.accretion_rate, 0.0);
}

/// Múltiples BlackHoles se persisten y restauran.
#[test]
fn agn_bhs_checkpoint_roundtrip() {
    let dir = tempdir().unwrap();
    let bhs = vec![
        BlackHole::new(Vec3::new(0.5, 0.5, 0.5), 1e5),
        BlackHole::new(Vec3::new(5.0, 5.0, 5.0), 2e6),
    ];
    save_agn_bhs(dir.path(), &bhs);
    let restored = load_agn_bhs(dir.path());
    assert_eq!(restored.len(), 2);
    assert!((restored[0].mass - 1e5).abs() < 1.0);
    assert!((restored[1].mass - 2e6).abs() < 1.0);
    assert!((restored[0].pos.x - 0.5).abs() < 1e-14);
    assert!((restored[1].pos.z - 5.0).abs() < 1e-14);
}

/// ChemState se serializa y deserializa correctamente (Serde derive Phase 106).
#[test]
fn chem_state_serde_roundtrip() {
    let cs = ChemState::neutral();
    let json = serde_json::to_string(&cs).unwrap();
    assert!(json.contains("x_hi"), "debe tener campo x_hi: {json}");
    let restored: ChemState = serde_json::from_str(&json).unwrap();
    assert!((restored.x_hi - cs.x_hi).abs() < 1e-15);
    assert!((restored.x_hii - cs.x_hii).abs() < 1e-15);
}

/// Vector de ChemStates se persiste y restaura.
#[test]
fn chem_states_checkpoint_roundtrip() {
    let dir = tempdir().unwrap();
    let mut cs0 = ChemState::neutral();
    cs0.x_hii = 0.5;
    cs0.x_hi = 0.5;
    let cs1 = ChemState::neutral();
    let states = vec![cs0.clone(), cs1.clone()];
    save_chem_states(dir.path(), &states);
    let restored = load_chem_states(dir.path());
    assert_eq!(restored.len(), 2);
    assert!(
        (restored[0].x_hii - 0.5).abs() < 1e-15,
        "x_hii debe ser 0.5"
    );
    assert!((restored[1].x_hii - cs1.x_hii).abs() < 1e-15);
}

/// SPH particle fields (internal_energy, smoothing_length) se preservan en
/// checkpoint JSONL — integración con Phase 105.
#[test]
fn sph_fields_in_checkpoint_jsonl() {
    let dir = tempdir().unwrap();
    let particles = vec![gas(0), gas(1), gas(2)];
    JsonlWriter
        .write(dir.path(), &particles, &prov(), &env())
        .unwrap();
    let data = JsonlReader.read(dir.path()).unwrap();
    assert_eq!(data.particles.len(), 3);
    for (orig, back) in particles.iter().zip(data.particles.iter()) {
        assert_eq!(back.ptype, ParticleType::Gas);
        assert!((back.internal_energy - orig.internal_energy).abs() < 1e-14);
        assert!((back.smoothing_length - orig.smoothing_length).abs() < 1e-14);
    }
}

/// Checkpoint completo: partículas SPH + BHs + chem states en el mismo directorio.
#[test]
fn full_checkpoint_all_state() {
    let dir = tempdir().unwrap();
    // 1. Partículas SPH
    let particles = vec![gas(0), gas(1)];
    JsonlWriter
        .write(dir.path(), &particles, &prov(), &env())
        .unwrap();
    // 2. AGN BHs
    let bhs = vec![BlackHole::new(Vec3::new(5.0, 5.0, 5.0), 1e6)];
    save_agn_bhs(dir.path(), &bhs);
    // 3. Chem states
    let mut cs = ChemState::neutral();
    cs.x_hii = 0.9;
    cs.x_hi = 0.1;
    let states = vec![cs.clone(), ChemState::neutral()];
    save_chem_states(dir.path(), &states);

    // Restaurar todo
    let p_data = JsonlReader.read(dir.path()).unwrap();
    let r_bhs = load_agn_bhs(dir.path());
    let r_chem = load_chem_states(dir.path());

    assert_eq!(p_data.particles.len(), 2);
    assert_eq!(p_data.particles[0].ptype, ParticleType::Gas);
    assert!((p_data.particles[0].internal_energy - 3.5).abs() < 1e-14);
    assert_eq!(r_bhs.len(), 1);
    assert!((r_bhs[0].mass - 1e6).abs() < 1.0);
    assert_eq!(r_chem.len(), 2);
    assert!((r_chem[0].x_hii - 0.9).abs() < 1e-15);
}
