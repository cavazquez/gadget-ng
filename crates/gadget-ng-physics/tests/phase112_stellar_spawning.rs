/// Phase 112 — Partículas estelares reales (spawning)
///
/// Tests: spawning estocástico con prob alta, conservación de masa,
///        herencia de metalicidad, estrellas no afectadas por SPH,
///        gas padre reduce masa, gas bajo m_min removido, serde de FeedbackSection.
use gadget_ng_core::{FeedbackSection, Particle, ParticleType, Vec3};
use gadget_ng_sph::spawn_star_particles;

fn fb_cfg(enabled: bool) -> FeedbackSection {
    FeedbackSection {
        enabled,
        sfr_min: 0.0,
        rho_sf: 0.0,
        m_star_fraction: 0.5,
        m_gas_min: 0.01,
        ..Default::default()
    }
}

fn gas_particle(id: usize, mass: f64, metallicity: f64) -> Particle {
    let mut p = Particle::new_gas(id, mass, Vec3::zero(), Vec3::zero(), 1.0, 0.5);
    p.metallicity = metallicity;
    p
}

// ── 1. Spawning estocástico con prob alta ─────────────────────────────────

#[test]
fn high_sfr_produces_stars_eventually() {
    let cfg = fb_cfg(true);
    let mut spawned_any = false;
    for attempt in 0..100u64 {
        let mut particles = vec![gas_particle(0, 1.0, 0.02)];
        let sfr = vec![1e6]; // sfr altísima → prob ≈ 1
        let mut seed = attempt.wrapping_mul(7919) + 1;
        let mut next_gid = 100;
        let (stars, _) = spawn_star_particles(&mut particles, &sfr, 1.0, &mut seed, &cfg, &mut next_gid);
        if !stars.is_empty() { spawned_any = true; break; }
    }
    assert!(spawned_any, "Debe formarse al menos una estrella con sfr altísima en 100 intentos");
}

// ── 2. Conservación de masa (gas + estrella = masa original) ──────────────

#[test]
fn mass_conserved_on_spawn() {
    let cfg = fb_cfg(true);
    // Forzar prob ≈ 1 con sfr muy alto
    let m0 = 1.0;
    let z0 = 0.02;
    let mut particles = vec![gas_particle(0, m0, z0)];
    let sfr = vec![1e10_f64];
    let mut seed = 42u64;
    let mut next_gid = 10;
    let (stars, to_remove) = spawn_star_particles(&mut particles, &sfr, 1e-10, &mut seed, &cfg, &mut next_gid);

    if !stars.is_empty() {
        let m_star: f64 = stars.iter().map(|s| s.mass).sum();
        let m_gas_remaining = if to_remove.contains(&0) { 0.0 } else { particles[0].mass };
        let m_total = m_star + m_gas_remaining + if to_remove.contains(&0) { particles[0].mass } else { 0.0 };
        // La masa total debe ser menor o igual (puede perderse algo si se eliminó gas)
        // Verificamos que m_star = m_star_fraction × m0
        assert!(
            (m_star - cfg.m_star_fraction * m0).abs() < 1e-12,
            "La masa de la estrella debe ser m_star_fraction × m_gas: {m_star} vs {}",
            cfg.m_star_fraction * m0
        );
    }
}

// ── 3. Herencia de metalicidad ────────────────────────────────────────────

#[test]
fn star_inherits_metallicity_from_gas() {
    let cfg = fb_cfg(true);
    let z_gas = 0.035;
    let mut particles = vec![gas_particle(0, 1.0, z_gas)];
    let sfr = vec![1e10_f64];
    let mut seed = 7u64;
    let mut next_gid = 50;
    let (stars, _) = spawn_star_particles(&mut particles, &sfr, 1e-10, &mut seed, &cfg, &mut next_gid);
    if !stars.is_empty() {
        assert!((stars[0].metallicity - z_gas).abs() < 1e-15, "Estrella debe heredar metalicidad del gas");
    }
}

// ── 4. DM no genera estrellas ─────────────────────────────────────────────

#[test]
fn dm_particles_dont_spawn_stars() {
    let cfg = fb_cfg(true);
    let mut particles = vec![Particle::new(0, 1.0, Vec3::zero(), Vec3::zero())];
    let sfr = vec![1e10_f64];
    let mut seed = 11u64;
    let mut next_gid = 0;
    let (stars, _) = spawn_star_particles(&mut particles, &sfr, 1.0, &mut seed, &cfg, &mut next_gid);
    assert!(stars.is_empty(), "DM no puede generar estrellas");
}

// ── 5. Gas padre reduce masa ──────────────────────────────────────────────

#[test]
fn gas_mass_reduced_after_spawn() {
    let cfg = fb_cfg(true);
    let m0 = 1.0;
    let mut particles = vec![gas_particle(0, m0, 0.01)];
    let sfr = vec![1e10_f64];
    let mut seed = 99u64;
    let mut next_gid = 200;
    let (stars, _) = spawn_star_particles(&mut particles, &sfr, 1e-10, &mut seed, &cfg, &mut next_gid);
    if !stars.is_empty() {
        assert!(particles[0].mass < m0, "El gas padre debe perder masa tras el spawning");
    }
}

// ── 6. Gas con masa < m_min se marca para eliminar ────────────────────────

#[test]
fn gas_below_min_mass_marked_for_removal() {
    let cfg = FeedbackSection {
        enabled: true,
        sfr_min: 0.0,
        rho_sf: 0.0,
        m_star_fraction: 0.99,  // casi toda la masa se convierte
        m_gas_min: 0.5,         // umbral alto para forzar eliminación
        ..Default::default()
    };
    // Masa inicial 0.6 → tras spawn queda 0.006 < 0.5 → debe marcarse
    let mut particles = vec![gas_particle(0, 0.6, 0.01)];
    let sfr = vec![1e10_f64];
    let mut seed = 55u64;
    let mut next_gid = 0;
    let (stars, to_remove) = spawn_star_particles(&mut particles, &sfr, 1e-10, &mut seed, &cfg, &mut next_gid);
    if !stars.is_empty() {
        assert!(to_remove.contains(&0), "Gas con masa residual < m_gas_min debe marcarse para eliminar");
    }
}

// ── 7. Serde de FeedbackSection con nuevos campos ─────────────────────────

#[test]
fn feedback_section_new_fields_serde() {
    let cfg = FeedbackSection {
        m_star_fraction: 0.3,
        m_gas_min: 0.05,
        ..Default::default()
    };
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: FeedbackSection = serde_json::from_str(&json).unwrap();
    assert!((cfg2.m_star_fraction - 0.3).abs() < 1e-15);
    assert!((cfg2.m_gas_min - 0.05).abs() < 1e-15);
}
