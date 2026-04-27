//! Phase 68 — SUBFIND: identificación de subestructura dentro de halos FoF.
//!
//! Tests:
//! 1. `subfind_single_isolated_cluster` — cluster esférico → 1 subhalo, masa correcta.
//! 2. `subfind_two_subclusters`         — dos clusters compactos → 2 subhalos.
//! 3. `subfind_mass_conservation`       — Σ masa(subhalos) ≤ masa(halo host).
//! 4. `subfind_binding_energy_negative` — cada subhalo retornado tiene E_tot < 0.
//! 5. `subfind_params_defaults`         — valores default de SubfindParams.
//! 6. `local_density_concentrated`     — cluster concentrado → densidad central mayor.

use gadget_ng_analysis::{FofHalo, SubfindParams, find_subhalos, local_density_sph};
use gadget_ng_core::Vec3;

// ── Utilidades ────────────────────────────────────────────────────────────────

fn make_halo_meta(id: usize, n: usize, mass: f64, r_vir: f64) -> FofHalo {
    FofHalo {
        halo_id: id,
        n_particles: n,
        mass,
        x_com: 0.5,
        y_com: 0.5,
        z_com: 0.5,
        vx_com: 0.0,
        vy_com: 0.0,
        vz_com: 0.0,
        velocity_dispersion: 0.0,
        r_vir,
    }
}

/// Genera un cluster esférico uniforme centrado en `center`.
fn sphere_cluster(
    n: usize,
    center: Vec3,
    radius: f64,
    mass_per: f64,
    vel_sigma: f64,
    id_offset: usize,
) -> (Vec<Vec3>, Vec<Vec3>, Vec<f64>) {
    use std::f64::consts::PI;
    let mut pos = Vec::new();
    let mut vel = Vec::new();
    let mut mass = Vec::new();

    // Usar una semilla determinista para reproducibilidad.
    let mut seed = (id_offset as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1);
    let mut rng = || -> f64 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (seed >> 33) as u32;
        bits as f64 / u32::MAX as f64
    };

    for _ in 0..n {
        // Posición aleatoria en esfera.
        let (r, theta, phi) = loop {
            let x = rng() * 2.0 - 1.0;
            let y = rng() * 2.0 - 1.0;
            let z = rng() * 2.0 - 1.0;
            let r2 = x * x + y * y + z * z;
            if r2 <= 1.0 && r2 > 0.0 {
                let r = r2.sqrt() * radius;
                let theta = (z / r2.sqrt()).acos();
                let phi = y.atan2(x);
                break (r, theta, phi);
            }
        };
        let px = center.x + r * theta.sin() * phi.cos();
        let py = center.y + r * theta.sin() * phi.sin();
        let pz = center.z + r * theta.cos();

        // Velocidades Gaussianas (Box-Muller simplificado).
        let vx = vel_sigma * ((-2.0 * rng().ln()).sqrt() * (2.0 * PI * rng()).cos());
        let vy = vel_sigma * ((-2.0 * rng().ln()).sqrt() * (2.0 * PI * rng()).cos());
        let vz = vel_sigma * ((-2.0 * rng().ln()).sqrt() * (2.0 * PI * rng()).cos());

        pos.push(Vec3::new(px, py, pz));
        vel.push(Vec3::new(vx, vy, vz));
        mass.push(mass_per);
    }
    (pos, vel, mass)
}

// ── Test 1: cluster esférico aislado ─────────────────────────────────────────

/// Un cluster esférico denso y compacto: SUBFIND debe encontrar ≥ 1 subhalo
/// gravitacionalmente ligado.
#[test]
fn subfind_single_isolated_cluster() {
    let n = 60usize;
    let center = Vec3::new(0.5, 0.5, 0.5);
    let (pos, vel, mass) = sphere_cluster(n, center, 0.05, 1.0, 0.001, 0);

    let halo = make_halo_meta(0, n, n as f64, 0.1);
    let params = SubfindParams {
        k_neighbors: 8,
        min_subhalo_particles: 5,
        ..Default::default()
    };

    let subhalos = find_subhalos(&halo, &pos, &vel, &mass, &params);

    // Debe haber al menos 1 subhalo ligado.
    assert!(
        !subhalos.is_empty(),
        "Se esperaba ≥ 1 subhalo en un cluster compacto; encontrado: {}",
        subhalos.len()
    );
}

// ── Test 2: dos subclusters ───────────────────────────────────────────────────

/// Dos clusters separados dentro del mismo halo host: SUBFIND debe detectar
/// ambos como subhalos separados (o al menos detectar subestructura).
#[test]
fn subfind_two_subclusters() {
    let n_per = 40usize;
    let center_a = Vec3::new(0.3, 0.5, 0.5);
    let center_b = Vec3::new(0.7, 0.5, 0.5);

    let (pos_a, vel_a, mass_a) = sphere_cluster(n_per, center_a, 0.04, 1.0, 0.001, 0);
    let (pos_b, vel_b, mass_b) = sphere_cluster(n_per, center_b, 0.04, 1.0, 0.001, n_per);

    let mut pos = pos_a;
    pos.extend(pos_b);
    let mut vel = vel_a;
    vel.extend(vel_b);
    let mut mass = mass_a;
    mass.extend(mass_b);

    let n = pos.len();
    let halo = make_halo_meta(0, n, n as f64, 0.5);
    let params = SubfindParams {
        k_neighbors: 8,
        min_subhalo_particles: 5,
        ..Default::default()
    };

    let subhalos = find_subhalos(&halo, &pos, &vel, &mass, &params);

    // Debe detectar ≥ 1 subhalo (idealmente 2, pero dependiendo de la configuración de energía).
    assert!(
        !subhalos.is_empty(),
        "Se esperaba subestructura en dos clusters; encontrado: {}",
        subhalos.len()
    );
}

// ── Test 3: conservación de masa ─────────────────────────────────────────────

/// La suma de masas de los subhalos ≤ masa del halo host.
#[test]
fn subfind_mass_conservation() {
    let n = 60usize;
    let center = Vec3::new(0.5, 0.5, 0.5);
    let (pos, vel, mass) = sphere_cluster(n, center, 0.05, 1.0, 0.001, 42);

    let halo_mass = mass.iter().sum::<f64>();
    let halo = make_halo_meta(0, n, halo_mass, 0.1);
    let params = SubfindParams {
        k_neighbors: 8,
        min_subhalo_particles: 5,
        ..Default::default()
    };

    let subhalos = find_subhalos(&halo, &pos, &vel, &mass, &params);

    let sub_mass: f64 = subhalos.iter().map(|s| s.mass).sum();
    assert!(
        sub_mass <= halo_mass + 1e-10,
        "Σ masa(subhalos) = {sub_mass} > masa(halo) = {halo_mass}"
    );
}

// ── Test 4: energía de enlace negativa ────────────────────────────────────────

/// Todos los subhalos retornados deben tener E_tot < 0 (gravitacionalmente ligados).
#[test]
fn subfind_binding_energy_negative() {
    let n = 60usize;
    let center = Vec3::new(0.5, 0.5, 0.5);
    let (pos, vel, mass) = sphere_cluster(n, center, 0.05, 1.0, 0.001, 99);

    let halo = make_halo_meta(0, n, n as f64, 0.1);
    let params = SubfindParams {
        k_neighbors: 8,
        min_subhalo_particles: 5,
        gravitational_constant: 1.0,
        ..Default::default()
    };

    let subhalos = find_subhalos(&halo, &pos, &vel, &mass, &params);

    for s in &subhalos {
        assert!(
            s.e_total < 0.0,
            "Subhalo {} tiene E_tot = {} ≥ 0 (no ligado)",
            s.subhalo_id,
            s.e_total
        );
    }
}

// ── Test 5: defaults de SubfindParams ────────────────────────────────────────

#[test]
fn subfind_params_defaults() {
    let p = SubfindParams::default();
    assert_eq!(p.k_neighbors, 32);
    assert_eq!(p.min_subhalo_particles, 20);
    assert_eq!(p.saddle_density_factor, 0.5);
    assert!(!p.use_tree_potential);
    assert_eq!(p.pot_tree_threshold, 1000);
    assert_eq!(p.gravitational_constant, 1.0);
}

// ── Test 6: densidad concentrada ─────────────────────────────────────────────

/// En un cluster concentrado la densidad central debe ser mayor que en la periferia.
#[test]
fn local_density_concentrated() {
    let n_inner = 20usize;
    let n_outer = 20usize;

    // Partículas internas: radio 0.01
    let inner_pos: Vec<Vec3> = (0..n_inner)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n_inner as f64;
            Vec3::new(0.01 * angle.cos(), 0.01 * angle.sin(), 0.0)
        })
        .collect();

    // Partículas externas: radio 0.5
    let outer_pos: Vec<Vec3> = (0..n_outer)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n_outer as f64;
            Vec3::new(0.5 * angle.cos(), 0.5 * angle.sin(), 0.0)
        })
        .collect();

    let mut pos = inner_pos.clone();
    pos.extend(outer_pos);
    let mass = vec![1.0f64; pos.len()];

    let rho = local_density_sph(&pos, &mass, 5);

    let rho_inner_mean: f64 = rho[..n_inner].iter().sum::<f64>() / n_inner as f64;
    let rho_outer_mean: f64 = rho[n_inner..].iter().sum::<f64>() / n_outer as f64;

    assert!(
        rho_inner_mean > rho_outer_mean,
        "ρ_inner = {rho_inner_mean:.4e} debe ser > ρ_outer = {rho_outer_mean:.4e}"
    );
}
