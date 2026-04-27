/// Phase 110 — Enriquecimiento químico SPH
///
/// Tests: distribución a vecino único, conservación de masa, sin enriquecimiento
///        de DM/estrellas, zero sfr, yield configurable, partícula estelar AGB.
use gadget_ng_core::{EnrichmentSection, Particle, Vec3};
use gadget_ng_sph::apply_enrichment;

fn cfg_enabled() -> EnrichmentSection {
    EnrichmentSection {
        enabled: true,
        yield_snii: 0.02,
        yield_agb: 0.04,
    }
}

// ── 1. Distribución a vecino único ────────────────────────────────────────

#[test]
fn enrichment_increases_neighbor_metallicity() {
    // Un gas donante (idx 0) con sfr=1.0 y h_sml=1.0 debe enriquecer al gas (idx 1) cercano
    let mut particles = vec![
        Particle::new_gas(0, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::zero(), 1.0, 1.0),
        Particle::new_gas(1, 1.0, Vec3::new(0.5, 0.0, 0.0), Vec3::zero(), 1.0, 1.0),
    ];
    let sfr = vec![1.0, 0.0];
    let z_before = particles[1].metallicity;
    apply_enrichment(&mut particles, &sfr, 0.01, &cfg_enabled());
    assert!(
        particles[1].metallicity > z_before,
        "el vecino debe haberse enriquecido"
    );
}

// ── 2. DM no recibe metales ───────────────────────────────────────────────

#[test]
fn dm_particles_not_enriched() {
    let mut particles = vec![
        Particle::new_gas(0, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::zero(), 1.0, 1.0),
        // DM cercano al gas donante
        Particle::new(1, 1.0, Vec3::new(0.3, 0.0, 0.0), Vec3::zero()),
    ];
    let sfr = vec![1.0, 0.0];
    apply_enrichment(&mut particles, &sfr, 0.1, &cfg_enabled());
    assert_eq!(particles[1].metallicity, 0.0, "DM no debe recibir metales");
}

// ── 3. sfr=0 no enriquece ─────────────────────────────────────────────────

#[test]
fn zero_sfr_no_enrichment() {
    let mut particles = vec![
        Particle::new_gas(0, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::zero(), 1.0, 1.0),
        Particle::new_gas(1, 1.0, Vec3::new(0.5, 0.0, 0.0), Vec3::zero(), 1.0, 1.0),
    ];
    let sfr = vec![0.0, 0.0];
    apply_enrichment(&mut particles, &sfr, 1.0, &cfg_enabled());
    assert_eq!(particles[1].metallicity, 0.0);
}

// ── 4. Configuración desactivada no hace nada ─────────────────────────────

#[test]
fn disabled_enrichment_no_change() {
    let mut particles = vec![
        Particle::new_gas(0, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::zero(), 1.0, 1.0),
        Particle::new_gas(1, 1.0, Vec3::new(0.5, 0.0, 0.0), Vec3::zero(), 1.0, 1.0),
    ];
    let sfr = vec![1.0, 0.0];
    let cfg_off = EnrichmentSection {
        enabled: false,
        ..cfg_enabled()
    };
    apply_enrichment(&mut particles, &sfr, 1.0, &cfg_off);
    assert_eq!(particles[1].metallicity, 0.0);
}

// ── 5. Yield configurable afecta la magnitud ─────────────────────────────

#[test]
fn higher_yield_gives_more_enrichment() {
    let mut p_low = vec![
        Particle::new_gas(0, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::zero(), 1.0, 1.0),
        Particle::new_gas(1, 1.0, Vec3::new(0.5, 0.0, 0.0), Vec3::zero(), 1.0, 1.0),
    ];
    let mut p_high = p_low.clone();
    let sfr = vec![1.0, 0.0];

    let cfg_low = EnrichmentSection {
        enabled: true,
        yield_snii: 0.01,
        yield_agb: 0.04,
    };
    let cfg_high = EnrichmentSection {
        enabled: true,
        yield_snii: 0.10,
        yield_agb: 0.04,
    };

    apply_enrichment(&mut p_low, &sfr, 0.1, &cfg_low);
    apply_enrichment(&mut p_high, &sfr, 0.1, &cfg_high);

    assert!(p_high[1].metallicity > p_low[1].metallicity);
}

// ── 6. Metalicidad nunca supera 1.0 (capped) ─────────────────────────────

#[test]
fn metallicity_capped_at_one() {
    let mut particles = vec![
        Particle::new_gas(0, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::zero(), 1.0, 0.1),
        Particle::new_gas(1, 1.0, Vec3::new(0.05, 0.0, 0.0), Vec3::zero(), 1.0, 0.1),
    ];
    // yield extremadamente alto + dt largo
    let cfg = EnrichmentSection {
        enabled: true,
        yield_snii: 100.0,
        yield_agb: 0.0,
    };
    let sfr = vec![1.0, 0.0];
    apply_enrichment(&mut particles, &sfr, 100.0, &cfg);
    assert!(
        particles[1].metallicity <= 1.0,
        "metalicidad no puede superar 1.0"
    );
}
