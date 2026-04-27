//! Phase 61 — FoF paralelo MPI.
//!
//! Verifica que `find_halos_parallel` con `SerialRuntime` (P=1) produce el mismo
//! catálogo que el FoF serial estándar, y que la función auxiliar
//! `find_halos_combined` recupera correctamente halos que cruzan una frontera artificial.

use gadget_ng_analysis::find_halos_parallel;
use gadget_ng_analysis::fof::{find_halos, find_halos_combined};
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_parallel::{SerialRuntime, sfc::SfcDecomposition};

fn skip() -> bool {
    std::env::var("PHASE61_SKIP")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Construye un conjunto de partículas en lattice perturbado y retorna posiciones.
fn make_lattice(n_side: usize, box_size: f64, seed: u64) -> Vec<Particle> {
    let n = n_side * n_side * n_side;
    let step = box_size / n_side as f64;
    let mut lcg = seed;
    let mut particles = Vec::with_capacity(n);
    for ix in 0..n_side {
        for iy in 0..n_side {
            for iz in 0..n_side {
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let dx = ((lcg >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * step * 0.1;
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let dy = ((lcg >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * step * 0.1;
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let dz = ((lcg >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * step * 0.1;
                let x = (ix as f64 * step + dx).rem_euclid(box_size);
                let y = (iy as f64 * step + dy).rem_euclid(box_size);
                let z = (iz as f64 * step + dz).rem_euclid(box_size);
                particles.push(Particle::new(
                    particles.len(),
                    1.0,
                    Vec3::new(x, y, z),
                    Vec3::zero(),
                ));
            }
        }
    }
    particles
}

/// Construye un cluster compacto centrado en (cx, cy, cz).
fn cluster(n: usize, cx: f64, cy: f64, cz: f64, radius: f64, seed: u64) -> Vec<Particle> {
    let mut lcg = seed;
    (0..n)
        .map(|i| {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let dx = ((lcg >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2.0 * radius;
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let dy = ((lcg >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2.0 * radius;
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let dz = ((lcg >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2.0 * radius;
            Particle::new(i, 1.0, Vec3::new(cx + dx, cy + dy, cz + dz), Vec3::zero())
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────────

/// Con SerialRuntime (P=1) find_halos_parallel debe producir el mismo número de
/// halos y la misma masa total que find_halos serial.
#[test]
fn phase61_vs_serial_p1() {
    if skip() {
        return;
    }

    let box_size = 1.0_f64;
    let particles = make_lattice(8, box_size, 42);
    let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let velocities: Vec<Vec3> = particles.iter().map(|p| p.velocity).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();

    let decomp = SfcDecomposition::build(&positions, box_size, 1);
    let rt = SerialRuntime;

    let halos_serial = find_halos(&positions, &velocities, &masses, box_size, 0.2, 5, 0.0);
    let halos_parallel = find_halos_parallel(&particles, &rt, &decomp, box_size, 0.2, 5, 0.0);

    assert_eq!(
        halos_serial.len(),
        halos_parallel.len(),
        "número de halos debe coincidir con P=1"
    );

    let mass_serial: f64 = halos_serial.iter().map(|h| h.mass).sum();
    let mass_parallel: f64 = halos_parallel.iter().map(|h| h.mass).sum();
    let rel_err = (mass_serial - mass_parallel).abs() / mass_serial.max(1e-30);
    assert!(
        rel_err < 1e-10,
        "masa total debe ser idéntica: serial={mass_serial}, paralelo={mass_parallel}"
    );
}

/// find_halos_combined recupera un halo que cruza la frontera artificial entre
/// partículas locales y halos recibidos.
#[test]
fn phase61_cross_boundary_halo_recovered() {
    if skip() {
        return;
    }

    let box_size = 1.0_f64;

    // Cluster compacto centrado en x=0.5, dividido en dos mitades.
    // "Local": mitad izquierda (x < 0.5).
    // "Halos": mitad derecha (x >= 0.5).
    let left_cluster = cluster(10, 0.47, 0.5, 0.5, 0.02, 100);
    let right_cluster = cluster(10, 0.53, 0.5, 0.5, 0.02, 200);

    let n_local = left_cluster.len();
    let all: Vec<Particle> = left_cluster.into_iter().chain(right_cluster).collect();
    let all_pos: Vec<Vec3> = all.iter().map(|p| p.position).collect();
    let all_vel: Vec<Vec3> = all.iter().map(|p| p.velocity).collect();
    let all_mass: Vec<f64> = all.iter().map(|p| p.mass).collect();

    // Con FoF combinado, el halo cruza la frontera y debe encontrarse entero.
    let halos_combined = find_halos_combined(
        &all_pos, &all_vel, &all_mass, n_local, box_size, 0.2, 5, 0.0,
    );

    // Debe haber exactamente 1 halo con al menos 10 partículas (locales + halos).
    assert_eq!(
        halos_combined.len(),
        1,
        "debe encontrarse 1 halo cross-boundary"
    );
    assert!(
        halos_combined[0].n_particles >= 10,
        "halo debe incluir partículas de ambas mitades: n={}",
        halos_combined[0].n_particles
    );
}

/// La suma de masas de halos encontrados por find_halos_parallel debe ser ≤ la
/// masa total de las partículas (conservación: halos pueden no incluir todas las
/// partículas de campo si min_particles > 1).
#[test]
fn phase61_mass_conservation() {
    if skip() {
        return;
    }

    let box_size = 1.0_f64;

    // Dos clusters bien separados + partículas de campo.
    let mut particles = cluster(20, 0.25, 0.25, 0.25, 0.04, 1);
    particles.extend(cluster(20, 0.75, 0.75, 0.75, 0.04, 2));
    // 10 partículas de campo distribuidas uniformemente.
    particles.extend(make_lattice(2, box_size, 99).into_iter().take(8));

    let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let decomp = SfcDecomposition::build(&positions, box_size, 1);
    let rt = SerialRuntime;

    let halos = find_halos_parallel(&particles, &rt, &decomp, box_size, 0.2, 10, 0.0);
    let halo_mass: f64 = halos.iter().map(|h| h.mass).sum();
    let total_mass: f64 = particles.iter().map(|p| p.mass).sum();

    assert!(
        halo_mass <= total_mass + 1e-10,
        "masa de halos ({halo_mass}) no puede superar la masa total ({total_mass})"
    );
    assert_eq!(halos.len(), 2, "deben encontrarse exactamente 2 halos");
}
