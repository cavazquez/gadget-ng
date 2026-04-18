//! Tests de geometría 3D periódica para el halo volumétrico (Fase 22).
//!
//! Cubre:
//! 1. `min_dist2_to_aabb_3d_trivial` — punto dentro del AABB → distancia 0
//! 2. `min_dist2_to_aabb_3d_along_x` — partícula en x=0.95 vs AABB [0,0.5)
//! 3. `min_dist2_to_aabb_3d_along_y` — idem en y
//! 4. `min_dist2_to_aabb_3d_along_z` — idem en z
//! 5. `min_dist2_to_aabb_3d_diagonal_xyz` — partícula en (0.95,0.95,0.95) vs AABB [0,0.5)³
//! 6. `halo_1d_misses_diagonal_periodic` — halo 1D-z excluye partícula diagonal
//! 7. `halo_3d_catches_diagonal_periodic` — halo 3D incluye partícula diagonal
//! 8. `compute_aabb_3d_correctness` — AABB real de un conjunto de partículas

use gadget_ng_parallel::halo3d::{
    compute_aabb_3d, is_in_periodic_halo, min_dist2_to_aabb_3d_periodic, Aabb3,
};
use gadget_ng_core::{Particle, Vec3};

fn make_aabb(xlo: f64, xhi: f64, ylo: f64, yhi: f64, zlo: f64, zhi: f64) -> Aabb3 {
    Aabb3 {
        lo: [xlo, ylo, zlo],
        hi: [xhi, yhi, zhi],
    }
}

fn make_particle(x: f64, y: f64, z: f64) -> Particle {
    Particle {
        position: Vec3::new(x, y, z),
        velocity: Vec3::zero(),
        acceleration: Vec3::zero(),
        mass: 1.0,
        global_id: 0,
    }
}

// ── Test 1: punto dentro del AABB → distancia 0 ───────────────────────────────

#[test]
fn min_dist2_to_aabb_3d_trivial() {
    let aabb = make_aabb(0.0, 0.5, 0.0, 0.5, 0.0, 0.5);
    let p = [0.25, 0.25, 0.25];
    let d2 = min_dist2_to_aabb_3d_periodic(p, &aabb, 1.0);
    assert!(
        d2 < 1e-14,
        "punto en interior del AABB debe tener distancia 0, got {d2}"
    );
}

// ── Test 2: distancia periódica en x ─────────────────────────────────────────

#[test]
fn min_dist2_to_aabb_3d_along_x() {
    // p en x=0.95, AABB con xlo=0.0, xhi=0.5.
    // CoM_x = 0.25; min_image(0.25 - 0.95, 1) = min_image(-0.7, 1) = 0.3
    // excess_x = |0.3| - 0.25 = 0.05 → dist2_x = 0.0025
    // y, z: p centrado → sin exceso.
    let aabb = make_aabb(0.0, 0.5, 0.0, 0.5, 0.0, 0.5);
    let p = [0.95, 0.25, 0.25];
    let d2 = min_dist2_to_aabb_3d_periodic(p, &aabb, 1.0);
    let expected = 0.05_f64.powi(2);
    assert!(
        (d2 - expected).abs() < 1e-10,
        "distancia x periódica: d2={d2:.4e} vs expected={expected:.4e}"
    );
    assert!(
        is_in_periodic_halo(p, &aabb, 0.1, 1.0),
        "con r_cut=0.1 la partícula x=0.95 debe estar en el halo de AABB [0,0.5)"
    );
}

// ── Test 3: distancia periódica en y ─────────────────────────────────────────

#[test]
fn min_dist2_to_aabb_3d_along_y() {
    let aabb = make_aabb(0.0, 0.5, 0.0, 0.5, 0.0, 0.5);
    let p = [0.25, 0.95, 0.25];
    let d2 = min_dist2_to_aabb_3d_periodic(p, &aabb, 1.0);
    let expected = 0.05_f64.powi(2);
    assert!(
        (d2 - expected).abs() < 1e-10,
        "distancia y periódica: d2={d2:.4e}"
    );
    assert!(
        is_in_periodic_halo(p, &aabb, 0.1, 1.0),
        "con r_cut=0.1 la partícula y=0.95 debe estar en el halo"
    );
}

// ── Test 4: distancia periódica en z ─────────────────────────────────────────

#[test]
fn min_dist2_to_aabb_3d_along_z() {
    let aabb = make_aabb(0.0, 0.5, 0.0, 0.5, 0.0, 0.5);
    let p = [0.25, 0.25, 0.95];
    let d2 = min_dist2_to_aabb_3d_periodic(p, &aabb, 1.0);
    let expected = 0.05_f64.powi(2);
    assert!(
        (d2 - expected).abs() < 1e-10,
        "distancia z periódica: d2={d2:.4e}"
    );
    assert!(
        is_in_periodic_halo(p, &aabb, 0.1, 1.0),
        "con r_cut=0.1 la partícula z=0.95 debe estar en el halo"
    );
}

// ── Test 5: distancia periódica diagonal (x+y+z) ─────────────────────────────

#[test]
fn min_dist2_to_aabb_3d_diagonal_xyz() {
    // p=(0.95, 0.95, 0.95), AABB=[0,0.5)³. CoM=(0.25,0.25,0.25), half=(0.25,0.25,0.25).
    // min_image(0.25-0.95, 1) = 0.3 por cada eje → excess = 0.05 por eje.
    // dist2 = 3 × 0.05² = 0.0075. dist ≈ 0.0866 < r_cut=0.1.
    let aabb = make_aabb(0.0, 0.5, 0.0, 0.5, 0.0, 0.5);
    let p = [0.95, 0.95, 0.95];
    let d2 = min_dist2_to_aabb_3d_periodic(p, &aabb, 1.0);
    let expected = 3.0 * 0.05_f64.powi(2);
    assert!(
        (d2 - expected).abs() < 1e-10,
        "diagonal periódico: d2={d2:.4e} vs expected={expected:.4e}"
    );
    let dist = d2.sqrt();
    assert!(
        dist < 0.1,
        "distancia diagonal periódica={dist:.4} debe ser < r_cut=0.1"
    );
}

// ── Test 6: halo 1D-z excluye partícula diagonal ─────────────────────────────
//
// Caso diagnóstico: demostración del gap del halo 1D para descomposición en octantes.

#[test]
fn halo_1d_misses_diagonal_periodic() {
    // Escenario: descomposición en 2 octantes (no Z-slab).
    // Rank 0 posee [0, 0.5)³ → my_z_lo=0.0, my_z_hi=0.5.
    // Rank 1 posee [0.5, 1)³ → z_lo_rank1=0.5.
    // Partícula de rank 1 en (0.95, 0.95, 0.95):
    //   halo 1D-z envía si z < z_lo_rank1 + r_cut = 0.5 + 0.1 = 0.6.
    //   z=0.95 > 0.6 → NO enviada. ✗
    let r_cut = 0.1_f64;
    let z_lo_rank1 = 0.5_f64;
    let z_particle = 0.95_f64;
    let halo_1d_threshold = z_lo_rank1 + r_cut; // = 0.6

    let included_by_1d_halo = z_particle < halo_1d_threshold;
    assert!(
        !included_by_1d_halo,
        "halo 1D-z DEBE excluir partícula con z={z_particle} (umbral={halo_1d_threshold})"
    );

    // Pero la distancia 3D periódica SÍ es < r_cut → hay una interacción perdida.
    let aabb_rank0 = make_aabb(0.0, 0.5, 0.0, 0.5, 0.0, 0.5);
    let p = [0.95, 0.95, 0.95];
    let d3d = min_dist2_to_aabb_3d_periodic(p, &aabb_rank0, 1.0).sqrt();
    assert!(
        d3d < r_cut,
        "distancia 3D periódica={d3d:.4} debe ser < r_cut={r_cut}: la interacción existe pero el halo 1D la omite"
    );
}

// ── Test 7: halo 3D sí captura la partícula diagonal ─────────────────────────

#[test]
fn halo_3d_catches_diagonal_periodic() {
    let aabb_rank0 = make_aabb(0.0, 0.5, 0.0, 0.5, 0.0, 0.5);
    let p = [0.95, 0.95, 0.95];
    let r_cut = 0.1_f64;

    assert!(
        is_in_periodic_halo(p, &aabb_rank0, r_cut, 1.0),
        "halo 3D periódico DEBE incluir partícula diagonal (0.95,0.95,0.95) con r_cut={r_cut}"
    );

    // Consistencia: la partícula excluida por el halo 1D está en el halo 3D.
    // Este test documenta la mejora de Fase 22 sobre Fase 21 para octantes/SFC.
    let r_cut_too_small = 0.08_f64; // sqrt(3*0.05²) ≈ 0.0866 > 0.08 → fuera del halo
    assert!(
        !is_in_periodic_halo(p, &aabb_rank0, r_cut_too_small, 1.0),
        "con r_cut={r_cut_too_small} < dist_diagonal={:.4}, la partícula NO debe estar en el halo",
        3.0_f64.sqrt() * 0.05
    );
}

// ── Test 8: compute_aabb_3d con conjunto conocido de partículas ───────────────

#[test]
fn compute_aabb_3d_correctness() {
    let particles = vec![
        make_particle(0.1, 0.2, 0.3),
        make_particle(0.8, 0.5, 0.1),
        make_particle(0.3, 0.9, 0.7),
        make_particle(0.0, 0.0, 0.0),
        make_particle(1.0, 1.0, 1.0),
    ];

    let aabb = compute_aabb_3d(&particles);

    // xlo=0.0, xhi=1.0
    assert!((aabb.lo[0] - 0.0).abs() < 1e-14, "xlo correcto");
    assert!((aabb.hi[0] - 1.0).abs() < 1e-14, "xhi correcto");
    // ylo=0.0, yhi=1.0
    assert!((aabb.lo[1] - 0.0).abs() < 1e-14, "ylo correcto");
    assert!((aabb.hi[1] - 1.0).abs() < 1e-14, "yhi correcto");
    // zlo=0.0, zhi=1.0
    assert!((aabb.lo[2] - 0.0).abs() < 1e-14, "zlo correcto");
    assert!((aabb.hi[2] - 1.0).abs() < 1e-14, "zhi correcto");

    // Verificar que el AABB contiene todas las partículas.
    for p in &particles {
        let pos = [p.position.x, p.position.y, p.position.z];
        for k in 0..3 {
            assert!(pos[k] >= aabb.lo[k] - 1e-12, "posición debajo del AABB");
            assert!(pos[k] <= aabb.hi[k] + 1e-12, "posición encima del AABB");
        }
    }

    // AABB vacía.
    let empty = compute_aabb_3d(&[]);
    assert!(!empty.is_valid(), "AABB de slice vacío debe ser inválida");
}

// ── Bonus: AABB rectangular (semiejes distintos) ──────────────────────────────

#[test]
fn rectangular_aabb_periodic_distance() {
    // AABB estrecha en y: [0.0,1.0] × [0.4,0.6] × [0.0,1.0].
    // Para una partícula en y=0.05, la distancia periódica al borde ylo=0.4:
    // CoM_y = 0.5, half_y = 0.1.
    // min_image(0.5 - 0.05, 1) = 0.45 → excess_y = |0.45| - 0.1 = 0.35.
    // Pero la imagen periódica del CoM más cercana: 0.5 - 1 = -0.5.
    // min_image(0.5 - 0.05, 1) = 0.45. |0.45| - 0.10 = 0.35.
    // Distancia al borde ylo=0.4: |y - ylo|_periódico. Con y=0.05:
    //   directo: 0.4 - 0.05 = 0.35 (fuera desde abajo).
    //   periódico desde yhi=0.6: L - y + yhi = 1 - 0.05 + 0.6? No, usamos min_image sobre el CoM.
    // El algoritmo usa CoM=(0.5), min_image(0.5 - 0.05, 1) = 0.45, excess = 0.45 - 0.1 = 0.35.
    // Esto equivale a la distancia directa al borde más cercano desde afuera.
    let aabb = make_aabb(0.0, 1.0, 0.4, 0.6, 0.0, 1.0);
    let p = [0.5, 0.05, 0.5]; // p_y=0.05 está fuera del AABB en y
    let d2 = min_dist2_to_aabb_3d_periodic(p, &aabb, 1.0);
    let expected_excess_y = 0.35_f64;
    assert!(
        (d2 - expected_excess_y * expected_excess_y).abs() < 1e-10,
        "AABB rectangular: d2={d2:.4e} expected {:.4e}",
        expected_excess_y * expected_excess_y
    );
}
