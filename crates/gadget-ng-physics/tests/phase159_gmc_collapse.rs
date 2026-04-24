//! Tests de integración — Phase 159: GMC collapse + IMF sampling Kroupa.

use gadget_ng_sph::gmc::{collapse_gmc, inject_sn_from_cluster, sample_stellar_mass, GmcCluster, KroupaImf};
use gadget_ng_core::{Particle, Vec3};

fn gas_particle_dense(rho: f64, x: f64, metallicity: f64) -> Particle {
    let h = (0.1_f64 / rho).cbrt().max(0.005);
    let mut p = Particle::new_gas(0, 0.1, Vec3 { x, y: 0.0, z: 0.0 }, Vec3::zero(), 1.0, h);
    p.metallicity = metallicity;
    p
}

fn sph_cfg() -> gadget_ng_core::config::SphSection {
    let mut cfg = gadget_ng_core::config::SphSection::default();
    cfg.gamma = 5.0 / 3.0;
    cfg
}

// T1: cluster masa > 0 para gas con densidad alta
#[test]
fn cluster_mass_positive() {
    let mut particles = vec![gas_particle_dense(100.0, 0.0, 0.02)];
    let clusters = collapse_gmc(&mut particles, 0.0, 0.01, 42);
    assert!(!clusters.is_empty(), "Debe formarse al menos un cúmulo con gas denso");
    for c in &clusters {
        assert!(c.mass_total > 0.0, "Masa del cúmulo debe ser > 0");
    }
}

// T2: IMF Kroupa muestrea masas en [M_min, M_max]
#[test]
fn imf_samples_in_range() {
    let imf = KroupaImf::default();
    for seed in 0..100u64 {
        let m = sample_stellar_mass(&imf, seed);
        assert!(m >= imf.m_min && m <= imf.m_max,
            "Masa muestreada fuera de rango: {} (seed={})", m, seed);
    }
}

// T3: masa de cúmulos ≤ masa inicial del gas
#[test]
fn mass_conserved_after_collapse() {
    let mut particles: Vec<Particle> = (0..10).map(|i| {
        gas_particle_dense(50.0 + i as f64 * 10.0, i as f64 * 0.5, 0.02)
    }).collect();
    let mass_gas_before: f64 = particles.iter().map(|p| p.mass).sum();
    let clusters = collapse_gmc(&mut particles, 0.0, 0.01, 99);
    let mass_clusters: f64 = clusters.iter().map(|c| c.mass_total).sum();
    assert!(mass_clusters <= mass_gas_before + 1e-10,
        "Masa de cúmulos ({:.4e}) no debe exceder masa del gas ({:.4e})", mass_clusters, mass_gas_before);
}

// T4: SN II solo de cúmulos jóvenes (age < 30 Myr)
#[test]
fn sn_only_from_young_clusters() {
    let cfg = sph_cfg();
    let mut gas = vec![gas_particle_dense(0.1, 0.3, 0.02)];
    let u_before = gas[0].internal_energy;

    let young = vec![GmcCluster { pos: [0.3, 0.0, 0.0], mass_total: 10.0, n_stars: 100, age_gyr: 0.0, metallicity: 0.02 }];
    inject_sn_from_cluster(&young, &mut gas, 0.01, &cfg);
    let u_after_young = gas[0].internal_energy;
    assert!(u_after_young >= u_before, "SN del cúmulo joven debe aumentar la energía interna");

    gas[0].internal_energy = u_before;
    let old = vec![GmcCluster { pos: [0.3, 0.0, 0.0], mass_total: 10.0, n_stars: 100, age_gyr: 0.1, metallicity: 0.02 }];
    inject_sn_from_cluster(&old, &mut gas, 0.01, &cfg);
    assert_eq!(gas[0].internal_energy, u_before, "Cúmulo viejo no debe inyectar SN II");
}

// T5: metalicidad heredada del gas progenitor
#[test]
fn metallicity_inherited() {
    let metallicity_gas = 0.03;
    let mut particles = vec![gas_particle_dense(100.0, 0.0, metallicity_gas)];
    let clusters = collapse_gmc(&mut particles, 0.0, 0.01, 7);
    for c in &clusters {
        assert_eq!(c.metallicity, metallicity_gas, "Metalicidad del cúmulo debe heredarse del gas");
    }
}

// T6: N=200 partículas sin panics
#[test]
fn gmc_collapse_n200_no_panic() {
    let cfg = sph_cfg();
    let mut particles: Vec<Particle> = (0..200).map(|i| {
        gas_particle_dense(10.0 + i as f64 * 5.0, i as f64 * 0.1, 0.01 + i as f64 * 0.0001)
    }).collect();
    let clusters = collapse_gmc(&mut particles, 0.0, 0.001, 42);
    inject_sn_from_cluster(&clusters, &mut particles, 0.001, &cfg);
    for p in &particles {
        assert!(p.internal_energy.is_finite(), "u debe ser finita");
    }
}
