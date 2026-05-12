//! Phase 180 — Pop III / primeras estrellas.

use gadget_ng_core::{Particle, PopIIISection, Vec3};
use gadget_ng_rt::{ChemState, F_D};
use gadget_ng_sph::{
    apply_pop_iii_pisn_feedback, form_pop_iii_clusters, is_pop_iii_candidate, sample_pop_iii_mass,
};

fn pop3_cfg() -> PopIIISection {
    PopIIISection {
        enabled: true,
        density_threshold: 1.0,
        max_temperature_k: 5.0e3,
        min_h2_fraction: 1e-4,
        min_hd_fraction: 1e-8,
        cluster_efficiency: 0.1,
        ..Default::default()
    }
}

fn dense_primordial_gas(z: f64) -> Particle {
    let mut p = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), 1.0, 0.1);
    p.metallicity = z;
    p.h2_fraction = 1e-3;
    p
}

#[test]
fn pop_iii_candidate_requires_primordial_molecular_gas() {
    let cfg = pop3_cfg();
    let p = dense_primordial_gas(0.0);
    assert!(is_pop_iii_candidate(&p, 1e-3, 0.0, &cfg, 5.0 / 3.0));
    assert!(is_pop_iii_candidate(&p, 0.0, 2e-8, &cfg, 5.0 / 3.0));
    assert!(!is_pop_iii_candidate(&p, 0.0, 0.0, &cfg, 5.0 / 3.0));

    let enriched = dense_primordial_gas(1e-2);
    assert!(!is_pop_iii_candidate(&enriched, 1e-3, 0.0, &cfg, 5.0 / 3.0));
}

#[test]
fn pop_iii_imf_is_top_heavy() {
    let cfg = pop3_cfg();
    let mut seed = 7;
    let masses: Vec<f64> = (0..128)
        .map(|_| sample_pop_iii_mass(&cfg, &mut seed))
        .collect();
    assert!(masses.iter().all(|&m| m >= cfg.imf_m_min_msun));
    let mean = masses.iter().sum::<f64>() / masses.len() as f64;
    assert!(
        mean > 20.0,
        "IMF Pop III debe ser top-heavy, mean={mean:.3}"
    );
}

#[test]
fn form_pop_iii_cluster_consumes_gas_mass() {
    let cfg = pop3_cfg();
    let mut particles = vec![dense_primordial_gas(0.0)];
    let before = particles[0].mass;
    let mut chem = ChemState::neutral();
    chem.x_h2 = 1e-3;
    chem.x_hi = 1.0 - 2.0 * chem.x_h2;
    chem.x_d = F_D;
    let clusters = form_pop_iii_clusters(
        &mut particles,
        &[(chem.x_h2, chem.x_hd)],
        &cfg,
        5.0 / 3.0,
        11,
    );
    assert_eq!(clusters.len(), 1);
    assert!(clusters[0].mass_total > 0.0);
    assert!(particles[0].mass < before);
}

#[test]
fn pisn_feedback_heats_and_enriches_neighbors() {
    let cfg = pop3_cfg();
    let mut particles = vec![dense_primordial_gas(0.0)];
    let clusters = form_pop_iii_clusters(&mut particles, &[(1e-3, 0.0)], &cfg, 5.0 / 3.0, 42);
    let u_before = particles[0].internal_energy;
    let z_before = particles[0].metallicity;
    let e = apply_pop_iii_pisn_feedback(&clusters, &mut particles, &cfg, None);
    assert!(e > 0.0);
    assert!(particles[0].internal_energy > u_before);
    assert!(particles[0].metallicity > z_before);
}

#[test]
fn pop_iii_config_serde_defaults_disabled() {
    let cfg: PopIIISection = toml::from_str("").expect("default PopIII config");
    assert!(!cfg.enabled);
    assert!(cfg.critical_metallicity > 0.0);
    assert!(cfg.imf_m_max_msun > cfg.imf_m_min_msun);
}
