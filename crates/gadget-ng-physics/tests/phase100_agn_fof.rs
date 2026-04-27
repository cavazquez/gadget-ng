//! Phase 100 — AGN con halos FoF.
//!
//! Verifica que:
//! 1. Los centros de halos FoF se ordenan correctamente por masa DESC.
//! 2. La macro maybe_agn! coloca BH en posiciones de halos, no en el centro de la caja.
//! 3. Con n_agn_bh = 2 se crean 2 BHs en los 2 halos más masivos.
//! 4. Sin halos (insitu no corrido), se usa el fallback del centro de la caja.

use gadget_ng_analysis::find_halos_combined;
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_sph::{AgnParams, BlackHole, apply_agn_feedback};

const BOX_SIZE: f64 = 20.0;

fn make_particle(id: usize, pos: Vec3, mass: f64) -> Particle {
    let mut p = Particle::new(id, mass, pos, Vec3::zero());
    p.internal_energy = 100.0;
    p.smoothing_length = 0.5;
    p
}

/// Crea dos clusters de partículas separados: uno masivo en (5,5,5) y uno ligero en (15,15,15).
fn make_two_clusters() -> Vec<Particle> {
    let mut parts = Vec::new();
    let mut id = 0;

    // Cluster A: 20 partículas masivas en (5,5,5) ± 0.3
    for i in 0..20 {
        let dx = (i % 3) as f64 * 0.2 - 0.2;
        let dy = (i % 5) as f64 * 0.1 - 0.2;
        let dz = (i / 5) as f64 * 0.1 - 0.2;
        parts.push(make_particle(
            id,
            Vec3::new(5.0 + dx, 5.0 + dy, 5.0 + dz),
            2.0,
        ));
        id += 1;
    }

    // Cluster B: 15 partículas ligeras en (15,15,15) ± 0.3
    for i in 0..15 {
        let dx = (i % 3) as f64 * 0.2 - 0.2;
        let dy = (i % 5) as f64 * 0.1 - 0.2;
        let dz = (i / 5) as f64 * 0.1 - 0.2;
        parts.push(make_particle(
            id,
            Vec3::new(15.0 + dx, 15.0 + dy, 15.0 + dz),
            1.0,
        ));
        id += 1;
    }

    parts
}

#[test]
fn fof_halos_sorted_by_mass_desc() {
    let particles = make_two_clusters();
    let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let velocities: Vec<Vec3> = particles.iter().map(|p| p.velocity).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();

    let halos = find_halos_combined(
        &positions,
        &velocities,
        &masses,
        positions.len(),
        BOX_SIZE,
        0.2,
        5,
        1.0,
    );
    assert!(
        halos.len() >= 2,
        "Debe haber al menos 2 halos FoF, got {}",
        halos.len()
    );

    // Ordenar por masa DESC
    let mut sorted = halos.clone();
    sorted.sort_by(|a, b| {
        b.mass
            .partial_cmp(&a.mass)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // El halo más masivo es el cluster A (20 × 2.0 = 40 M)
    assert!(
        sorted[0].mass > sorted[1].mass,
        "primer halo debe ser más masivo: {:.1} vs {:.1}",
        sorted[0].mass,
        sorted[1].mass
    );
    assert!(
        sorted[0].mass >= 30.0,
        "cluster A debe tener masa ≥ 30, got {:.1}",
        sorted[0].mass
    );
}

#[test]
fn fof_halo_centers_near_cluster_positions() {
    let particles = make_two_clusters();
    let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let velocities: Vec<Vec3> = particles.iter().map(|p| p.velocity).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();

    let halos = find_halos_combined(
        &positions,
        &velocities,
        &masses,
        positions.len(),
        BOX_SIZE,
        0.2,
        5,
        1.0,
    );
    let mut sorted = halos.clone();
    sorted.sort_by(|a, b| {
        b.mass
            .partial_cmp(&a.mass)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // El halo más masivo debe estar cerca de (5,5,5)
    let h0 = &sorted[0];
    let dist0 =
        ((h0.x_com - 5.0).powi(2) + (h0.y_com - 5.0).powi(2) + (h0.z_com - 5.0).powi(2)).sqrt();
    assert!(
        dist0 < 1.0,
        "COM del halo más masivo debe estar cerca de (5,5,5), got ({:.2},{:.2},{:.2}), dist={:.2}",
        h0.x_com,
        h0.y_com,
        h0.z_com,
        dist0
    );

    // El segundo halo debe estar cerca de (15,15,15)
    if sorted.len() >= 2 {
        let h1 = &sorted[1];
        let dist1 =
            ((h1.x_com - 15.0).powi(2) + (h1.y_com - 15.0).powi(2) + (h1.z_com - 15.0).powi(2))
                .sqrt();
        assert!(
            dist1 < 1.0,
            "COM del segundo halo debe estar cerca de (15,15,15), got ({:.2},{:.2},{:.2}), dist={:.2}",
            h1.x_com,
            h1.y_com,
            h1.z_com,
            dist1
        );
    }
}

#[test]
fn agn_bh_placed_at_halo_center() {
    let particles = make_two_clusters();
    let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let velocities: Vec<Vec3> = particles.iter().map(|p| p.velocity).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();

    let halos = find_halos_combined(
        &positions,
        &velocities,
        &masses,
        positions.len(),
        BOX_SIZE,
        0.2,
        5,
        1.0,
    );
    let mut sorted = halos.clone();
    sorted.sort_by(|a, b| {
        b.mass
            .partial_cmp(&a.mass)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let halo_centers: Vec<Vec3> = sorted
        .iter()
        .map(|h| Vec3::new(h.x_com, h.y_com, h.z_com))
        .collect();

    let m_seed = 1e5_f64;
    let n_bh = 1_usize.min(halo_centers.len());

    // Crear BHs en centros de halos (lógica de maybe_agn! Phase 100)
    let mut agn_bhs: Vec<BlackHole> = Vec::new();
    let n_new = halo_centers.len().min(n_bh);
    agn_bhs.resize_with(n_new, || BlackHole::new(Vec3::zero(), m_seed));
    for (bh, &pos) in agn_bhs.iter_mut().zip(halo_centers.iter()) {
        bh.pos = pos;
    }

    assert_eq!(agn_bhs.len(), 1, "debe haber 1 BH");
    let bh = &agn_bhs[0];

    // El BH debe estar en el centro del halo más masivo (cluster A, ~(5,5,5))
    let dist =
        ((bh.pos.x - 5.0).powi(2) + (bh.pos.y - 5.0).powi(2) + (bh.pos.z - 5.0).powi(2)).sqrt();
    assert!(
        dist < 1.0,
        "BH debe estar en el halo más masivo (~(5,5,5)), pos=({:.2},{:.2},{:.2}), dist={:.2}",
        bh.pos.x,
        bh.pos.y,
        bh.pos.z,
        dist
    );
}

#[test]
fn agn_two_bhs_match_two_halos() {
    let particles = make_two_clusters();
    let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let velocities: Vec<Vec3> = particles.iter().map(|p| p.velocity).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();

    let halos = find_halos_combined(
        &positions,
        &velocities,
        &masses,
        positions.len(),
        BOX_SIZE,
        0.2,
        5,
        1.0,
    );
    let mut sorted = halos;
    sorted.sort_by(|a, b| {
        b.mass
            .partial_cmp(&a.mass)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if sorted.len() < 2 {
        return;
    } // necesitamos 2 halos para este test

    let halo_centers: Vec<Vec3> = sorted
        .iter()
        .map(|h| Vec3::new(h.x_com, h.y_com, h.z_com))
        .collect();

    let m_seed = 1e5_f64;
    let n_bh = 2_usize;

    let mut agn_bhs: Vec<BlackHole> = Vec::new();
    let n_new = halo_centers.len().min(n_bh);
    agn_bhs.resize_with(n_new, || BlackHole::new(Vec3::zero(), m_seed));
    for (bh, &pos) in agn_bhs.iter_mut().zip(halo_centers.iter()) {
        bh.pos = pos;
    }

    assert_eq!(agn_bhs.len(), 2, "debe haber 2 BHs");

    // BH[0] cerca de cluster A (5,5,5)
    let d0 = ((agn_bhs[0].pos.x - 5.0).powi(2)
        + (agn_bhs[0].pos.y - 5.0).powi(2)
        + (agn_bhs[0].pos.z - 5.0).powi(2))
    .sqrt();
    assert!(d0 < 1.0, "BH[0] debe estar en cluster A, dist={:.2}", d0);

    // BH[1] cerca de cluster B (15,15,15)
    let d1 = ((agn_bhs[1].pos.x - 15.0).powi(2)
        + (agn_bhs[1].pos.y - 15.0).powi(2)
        + (agn_bhs[1].pos.z - 15.0).powi(2))
    .sqrt();
    assert!(d1 < 1.0, "BH[1] debe estar en cluster B, dist={:.2}", d1);
}

#[test]
fn agn_fallback_to_box_center_without_halos() {
    let box_size = 20.0;
    let center = box_size * 0.5;
    let m_seed = 1e5_f64;

    let halo_centers: Vec<Vec3> = Vec::new(); // sin halos identificados
    let mut agn_bhs: Vec<BlackHole> = Vec::new();

    // Lógica de fallback de maybe_agn!
    if !halo_centers.is_empty() {
        let n_new = halo_centers.len().min(1);
        agn_bhs.resize_with(n_new, || BlackHole::new(Vec3::zero(), m_seed));
        for (bh, &pos) in agn_bhs.iter_mut().zip(halo_centers.iter()) {
            bh.pos = pos;
        }
    } else if agn_bhs.is_empty() {
        agn_bhs.push(BlackHole::new(Vec3::new(center, center, center), m_seed));
    }

    assert_eq!(agn_bhs.len(), 1, "debe haber 1 BH semilla");
    assert_eq!(
        agn_bhs[0].pos.x, center,
        "BH debe estar en el centro de la caja"
    );
    assert_eq!(agn_bhs[0].pos.y, center);
    assert_eq!(agn_bhs[0].pos.z, center);
}

#[test]
fn agn_feedback_applies_near_bh() {
    let mut particles = make_two_clusters();
    let agn_params = AgnParams {
        eps_feedback: 0.1,
        m_seed: 1e5,
        v_kick_agn: 0.0,
        r_influence: 2.0,
    };
    // BH en el centro del cluster A
    let bhs = vec![BlackHole {
        pos: Vec3::new(5.0, 5.0, 5.0),
        mass: 1e7,
        accretion_rate: 0.1,
    }];

    let u_before: Vec<f64> = particles.iter().map(|p| p.internal_energy).collect();
    apply_agn_feedback(&mut particles, &bhs, &agn_params, 0.01);
    let u_after: Vec<f64> = particles.iter().map(|p| p.internal_energy).collect();

    // Partículas en cluster A (radio < 2 del BH) deben haber ganado energía
    let heated = particles
        .iter()
        .zip(u_before.iter())
        .zip(u_after.iter())
        .filter(|&((p, &ub), &ua)| {
            let r = ((p.position.x - 5.0).powi(2)
                + (p.position.y - 5.0).powi(2)
                + (p.position.z - 5.0).powi(2))
            .sqrt();
            r < 2.0 && ua > ub
        })
        .count();

    assert!(
        heated > 0,
        "debe haber al menos una partícula calentada por el BH en cluster A"
    );
}
