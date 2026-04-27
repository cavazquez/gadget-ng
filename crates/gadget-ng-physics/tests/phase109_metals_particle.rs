/// Phase 109 — Metales en Particle y ParticleType::Star
///
/// Tests: new_star constructor, metallicity serde, backward compat, is_star,
///        ptype variants, EnrichmentSection defaults.
use gadget_ng_core::{EnrichmentSection, Particle, ParticleType, Vec3};

fn pos() -> Vec3 {
    Vec3::new(1.0, 2.0, 3.0)
}
fn vel() -> Vec3 {
    Vec3::new(0.1, 0.2, 0.3)
}

// ── 1. new_star constructor ────────────────────────────────────────────────

#[test]
fn new_star_sets_ptype_star() {
    let s = Particle::new_star(42, 1.0, pos(), vel(), 0.02);
    assert_eq!(s.ptype, ParticleType::Star);
}

#[test]
fn new_star_stores_metallicity() {
    let s = Particle::new_star(0, 1.0, pos(), vel(), 0.05);
    assert!((s.metallicity - 0.05).abs() < 1e-15);
}

#[test]
fn new_star_stellar_age_zero() {
    let s = Particle::new_star(0, 1.0, pos(), vel(), 0.01);
    assert_eq!(s.stellar_age, 0.0);
}

// ── 2. is_star / is_gas ────────────────────────────────────────────────────

#[test]
fn is_star_and_is_gas_exclusive() {
    let star = Particle::new_star(0, 1.0, pos(), vel(), 0.0);
    let gas = Particle::new_gas(1, 1.0, pos(), vel(), 0.5, 0.1);
    let dm = Particle::new(2, 1.0, pos(), vel());

    assert!(star.is_star());
    assert!(!star.is_gas());

    assert!(!gas.is_star());
    assert!(gas.is_gas());

    assert!(!dm.is_star());
    assert!(!dm.is_gas());
}

// ── 3. ParticleType variants all exist ────────────────────────────────────

#[test]
fn particle_type_three_variants() {
    let types = [
        ParticleType::DarkMatter,
        ParticleType::Gas,
        ParticleType::Star,
    ];
    assert_eq!(types.len(), 3);
    // Cada variante es distinta
    assert_ne!(types[0], types[1]);
    assert_ne!(types[1], types[2]);
    assert_ne!(types[0], types[2]);
}

// ── 4. metallicity serde roundtrip ────────────────────────────────────────

#[test]
fn metallicity_serde_roundtrip() {
    let mut p = Particle::new_gas(0, 1.0, pos(), vel(), 1.0, 0.1);
    p.metallicity = 0.03;
    p.stellar_age = 2.5;

    let json = serde_json::to_string(&p).unwrap();
    let p2: Particle = serde_json::from_str(&json).unwrap();

    assert!((p2.metallicity - 0.03).abs() < 1e-15);
    assert!((p2.stellar_age - 2.5).abs() < 1e-15);
}

// ── 5. backward compat: missing metallicity/stellar_age defaults to 0 ─────

#[test]
fn backward_compat_missing_metal_fields() {
    // JSON sin los campos nuevos — deben caer a 0.0 via #[serde(default)]
    let json = r#"{
        "global_id": 7,
        "mass": 2.0,
        "px": 0.0, "py": 0.0, "pz": 0.0,
        "vx": 0.0, "vy": 0.0, "vz": 0.0
    }"#;

    // ParticleRecord en gadget-ng-io tiene el formato con px/py/pz
    // Aquí usamos Particle directamente con su formato serde
    let json2 = r#"{
        "global_id": 7,
        "mass": 2.0,
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0}
    }"#;
    let p: Particle = serde_json::from_str(json2).unwrap();
    assert_eq!(p.metallicity, 0.0);
    assert_eq!(p.stellar_age, 0.0);
}

// ── 6. EnrichmentSection defaults ─────────────────────────────────────────

#[test]
fn enrichment_section_defaults() {
    let cfg = EnrichmentSection::default();
    assert!(!cfg.enabled);
    assert!((cfg.yield_snii - 0.02).abs() < 1e-15);
    assert!((cfg.yield_agb - 0.04).abs() < 1e-15);
}

#[test]
fn enrichment_section_serde_roundtrip() {
    let cfg = EnrichmentSection {
        enabled: true,
        yield_snii: 0.035,
        yield_agb: 0.06,
    };
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: EnrichmentSection = serde_json::from_str(&json).unwrap();
    assert!(cfg2.enabled);
    assert!((cfg2.yield_snii - 0.035).abs() < 1e-15);
    assert!((cfg2.yield_agb - 0.06).abs() < 1e-15);
}
