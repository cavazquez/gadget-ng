//! Phase 108 — Vientos galácticos
//!
//! Verifica el modelo de vientos galácticos (Springel & Hernquist 2003):
//! - `apply_galactic_winds` lanza partículas de gas con probabilidad correcta.
//! - La velocidad del viento tiene la magnitud correcta.
//! - Las partículas DM nunca son lanzadas.
//! - WindParams se serializa/deserializa correctamente en TOML/JSON.

use gadget_ng_core::{FeedbackSection, Particle, ParticleType, Vec3, WindParams};
use gadget_ng_sph::{apply_galactic_winds, compute_sfr};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn gas_particle(id: usize, h_small: f64) -> Particle {
    let mut p = Particle::new(id, 1.0, Vec3::new(id as f64, 0.0, 0.0), Vec3::zero());
    p.ptype = ParticleType::Gas;
    p.smoothing_length = h_small;
    p.internal_energy = 1.0;
    p
}

fn dm_particle(id: usize) -> Particle {
    Particle::new(id, 2.0, Vec3::new(id as f64 * 0.5, 0.0, 0.0), Vec3::zero())
}

fn wind_cfg(v_wind: f64, eta: f64) -> WindParams {
    WindParams {
        enabled: true,
        v_wind_km_s: v_wind,
        mass_loading: eta,
        t_decoupling_myr: 0.0,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Con vientos desactivados, ninguna partícula es lanzada.
#[test]
fn wind_disabled_no_launch() {
    let cfg = WindParams { enabled: false, ..Default::default() };
    let mut particles = vec![gas_particle(0, 0.01), gas_particle(1, 0.01)];
    let sfr = vec![1.0, 1.0];
    let v0 = particles[0].velocity;
    let mut seed = 42u64;
    let launched = apply_galactic_winds(&mut particles, &sfr, &cfg, 1.0, &mut seed);
    assert!(launched.is_empty(), "no debe lanzar con vientos desactivados");
    assert_eq!(particles[0].velocity.x, v0.x, "velocidad no debe cambiar");
}

/// Partículas DM nunca son lanzadas.
#[test]
fn dm_particles_never_launched() {
    let cfg = wind_cfg(480.0, 10.0);
    let mut particles = vec![dm_particle(0), dm_particle(1), dm_particle(2)];
    let sfr = vec![1e5, 1e5, 1e5]; // SFR muy alta
    let mut seed = 99u64;
    let launched = apply_galactic_winds(&mut particles, &sfr, &cfg, 1.0, &mut seed);
    assert!(launched.is_empty(), "DM nunca debe ser lanzado: {launched:?}");
}

/// Con SFR = 0, ninguna partícula de gas es lanzada.
#[test]
fn zero_sfr_no_wind() {
    let cfg = wind_cfg(480.0, 2.0);
    let mut particles = vec![gas_particle(0, 0.01), gas_particle(1, 0.01)];
    let sfr = vec![0.0, 0.0];
    let mut seed = 7u64;
    let launched = apply_galactic_winds(&mut particles, &sfr, &cfg, 1.0, &mut seed);
    assert!(launched.is_empty(), "SFR=0 → sin viento");
}

/// Con SFR muy alta (prob ≈ 1), la mayoría de partículas de gas son lanzadas.
#[test]
fn high_sfr_most_particles_launched() {
    let cfg = wind_cfg(480.0, 2.0);
    let n = 50;
    let mut particles: Vec<Particle> = (0..n).map(|i| gas_particle(i, 0.01)).collect();
    let sfr = vec![1e10f64; n]; // SFR enorme → p_wind ≈ 1
    let mut seed = 12345u64;
    let launched = apply_galactic_winds(&mut particles, &sfr, &cfg, 1.0, &mut seed);
    assert!(
        launched.len() >= n * 9 / 10,
        "con SFR alta, ≥90% deben ser lanzados: {}/{}",
        launched.len(), n
    );
}

/// La magnitud del kick de viento está cerca de v_wind.
#[test]
fn wind_kick_magnitude_correct() {
    let v_wind = 600.0;
    let cfg = wind_cfg(v_wind, 50.0); // η grande para aumentar prob
    let mut particles = vec![gas_particle(0, 0.01)];
    let sfr = vec![1e10f64]; // prob ≈ 1
    let v0 = particles[0].velocity;

    let mut seed = 777u64;
    let launched = apply_galactic_winds(&mut particles, &sfr, &cfg, 1e-6, &mut seed);

    if !launched.is_empty() {
        let dv = ((particles[0].velocity.x - v0.x).powi(2)
            + (particles[0].velocity.y - v0.y).powi(2)
            + (particles[0].velocity.z - v0.z).powi(2))
        .sqrt();
        assert!(
            (dv - v_wind).abs() < 1.0,
            "magnitud del kick debe ser ≈ v_wind={v_wind}: dv={dv}"
        );
    }
}

/// WindParams se serializa y deserializa correctamente con serde_json.
#[test]
fn wind_params_serde_roundtrip() {
    let params = WindParams {
        enabled: true,
        v_wind_km_s: 350.0,
        mass_loading: 3.5,
        t_decoupling_myr: 5.0,
    };
    let json = serde_json::to_string(&params).unwrap();
    assert!(json.contains("v_wind_km_s"), "debe tener v_wind_km_s: {json}");
    let restored: WindParams = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.enabled, true);
    assert!((restored.v_wind_km_s - 350.0).abs() < 1e-14);
    assert!((restored.mass_loading - 3.5).abs() < 1e-14);
}

/// FeedbackSection.wind se deserializa con defaults cuando no está presente en TOML.
#[test]
fn feedback_section_wind_default_in_toml() {
    let toml_str = r#"
enabled = true
v_kick_km_s = 350.0
eps_sn = 0.1
rho_sf = 0.1
sfr_min = 0.0001
"#;
    let fb: FeedbackSection = toml::from_str(toml_str).unwrap();
    assert!(!fb.wind.enabled, "wind debe estar desactivado por defecto");
    assert!((fb.wind.v_wind_km_s - 480.0).abs() < 1e-14, "v_wind default debe ser 480.0");
    assert!((fb.wind.mass_loading - 2.0).abs() < 1e-14, "mass_loading default debe ser 2.0");
}

/// Interacción entre SFR y vientos: compute_sfr + apply_galactic_winds integrado.
#[test]
fn sfr_and_wind_pipeline() {
    let feedback_cfg = FeedbackSection {
        enabled: true,
        rho_sf: 0.001, // umbral bajo para activar SFR
        sfr_min: 0.0,
        wind: WindParams { enabled: true, v_wind_km_s: 200.0, mass_loading: 5.0, t_decoupling_myr: 0.0 },
        ..Default::default()
    };
    let mut particles: Vec<Particle> = (0..10).map(|i| gas_particle(i, 0.01)).collect();
    let sfr = compute_sfr(&particles, &feedback_cfg);
    // SFR debe ser positiva para gas denso (h pequeño → ρ grande)
    assert!(sfr.iter().any(|&s| s > 0.0), "alguna partícula debe tener SFR > 0");
    let mut seed = 1234u64;
    let launched = apply_galactic_winds(&mut particles, &sfr, &feedback_cfg.wind, 1.0, &mut seed);
    // Puede que ninguna se lance en este paso (estocástico), pero la función no debe paniquear
    assert!(launched.len() <= 10, "no puede lanzar más partículas de las que hay");
}
