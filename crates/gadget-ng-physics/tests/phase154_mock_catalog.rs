//! Tests de integración — Phase 154: Mock catalogues con efectos de selección.

use gadget_ng_analysis::fof::FofHalo;
use gadget_ng_analysis::mock_catalog::{
    angular_power_spectrum_cl, apparent_magnitude, build_mock_catalog, selection_flux_limit,
};
use gadget_ng_core::{Particle, Vec3};

fn dm_particle(x: f64, y: f64, z: f64) -> Particle {
    Particle::new(0, 1.0, Vec3 { x, y, z }, Vec3::zero())
}

fn make_halos() -> Vec<FofHalo> {
    vec![
        FofHalo {
            halo_id: 0,
            n_particles: 100,
            mass: 1e2,
            x_com: 25.0,
            y_com: 25.0,
            z_com: 25.0,
            vx_com: 0.0,
            vy_com: 0.0,
            vz_com: 0.0,
            velocity_dispersion: 100.0,
            r_vir: 1.0,
        },
        FofHalo {
            halo_id: 1,
            n_particles: 50,
            mass: 5e1,
            x_com: 75.0,
            y_com: 75.0,
            z_com: 75.0,
            vx_com: 0.0,
            vy_com: 0.0,
            vz_com: 0.0,
            velocity_dispersion: 50.0,
            r_vir: 0.7,
        },
        FofHalo {
            halo_id: 2,
            n_particles: 200,
            mass: 5e2,
            x_com: 50.0,
            y_com: 10.0,
            z_com: 30.0,
            vx_com: 0.0,
            vy_com: 0.0,
            vz_com: 0.0,
            velocity_dispersion: 200.0,
            r_vir: 2.0,
        },
    ]
}

// T1: catálogo no vacío con halos masivos
#[test]
fn catalog_not_empty() {
    let particles: Vec<Particle> = Vec::new();
    let halos = make_halos();
    let catalog = build_mock_catalog(&particles, &halos, 0.1, 0.3, 30.0);
    assert!(!catalog.is_empty(), "El catálogo debe tener galaxias");
}

// T2: magnitud aparente crece con z
#[test]
fn apparent_magnitude_grows_with_z() {
    let m_abs = -21.0;
    let m1 = apparent_magnitude(m_abs, 0.05, 0.3);
    let m2 = apparent_magnitude(m_abs, 0.5, 0.3);
    let m3 = apparent_magnitude(m_abs, 1.0, 0.3);
    assert!(m2 > m1, "m_app(z=0.5) debe ser mayor que m_app(z=0.05)");
    assert!(m3 > m2, "m_app(z=1.0) debe ser mayor que m_app(z=0.5)");
}

// T3: selección por flujo reduce el catálogo
#[test]
fn flux_limit_reduces_catalog() {
    let particles: Vec<Particle> = Vec::new();
    let halos = make_halos();
    let catalog_deep = build_mock_catalog(&particles, &halos, 0.5, 0.3, 30.0);
    let catalog_shallow = build_mock_catalog(&particles, &halos, 0.5, 0.3, 10.0);
    assert!(
        catalog_shallow.len() <= catalog_deep.len(),
        "Catálogo superficial debe tener <= galaxias que el profundo"
    );
}

// T4: SMHM trend correcto (halos más masivos → más estrellas)
#[test]
fn smhm_trend() {
    let particles: Vec<Particle> = Vec::new();
    let halos = make_halos();
    let catalog = build_mock_catalog(&particles, &halos, 0.1, 0.3, 30.0);
    if catalog.len() >= 2 {
        let mut sorted = catalog.clone();
        sorted.sort_by(|a, b| a.halo_mass.partial_cmp(&b.halo_mass).unwrap());
        let first = &sorted[0];
        let last = &sorted[sorted.len() - 1];
        assert!(
            last.stellar_mass >= first.stellar_mass,
            "Halos más masivos deben tener más masa estelar"
        );
    }
}

// T5: selection_flux_limit funciona correctamente
#[test]
fn flux_limit_selection_works() {
    assert!(selection_flux_limit(15.0, 20.0), "m<m_lim debe pasar");
    assert!(!selection_flux_limit(25.0, 20.0), "m>m_lim no debe pasar");
}

// T6: N=500 partículas sin panics
#[test]
fn mock_catalog_n500_no_panic() {
    let mut particles = Vec::new();
    for i in 0..500 {
        particles.push(dm_particle(
            (i % 100) as f64,
            ((i / 100) % 10) as f64,
            (i / 1000) as f64,
        ));
    }
    let halos = make_halos();
    let catalog = build_mock_catalog(&particles, &halos, 0.3, 0.3, 25.0);
    let _cl = angular_power_spectrum_cl(&catalog, 5, 100.0);
}
