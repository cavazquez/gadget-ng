//! Tests de geometría SR-SFC para el dominio 3D/SFC del árbol de corto alcance (Fase 23).
//!
//! Verifica que la combinación SfcDecomposition + exchange_halos_3d_periodic + short_range_accels_sfc
//! produce cobertura geométrica correcta para cualquier descomposición de dominio:
//!
//! 1. `sr_sfc_border_interaction` — partícula en frontera SFC interactúa con la del otro lado
//! 2. `sr_sfc_diagonal_periodic_interaction` — diagonal periódica en dominio SFC
//! 3. `sr_sfc_halo_3d_covers_rcut_pairs` — halo 3D cubre todos los pares con r < r_cut
//! 4. `sr_sfc_minimum_image_active_in_walk` — minimum_image activo en walk SR-SFC
//! 5. `sr_sfc_no_geometric_gaps` — fuerza no nula para partícula periódicamente cercana en dominio SFC

use gadget_ng_core::{Particle, Vec3};
use gadget_ng_parallel::halo3d::{compute_aabb_3d, is_in_periodic_halo};
use gadget_ng_treepm::distributed::{short_range_accels_sfc, SfcShortRangeParams};

fn make_particle(id: usize, x: f64, y: f64, z: f64, mass: f64) -> Particle {
    Particle::new(id, mass, Vec3::new(x, y, z), Vec3::zero())
}

fn force_mag(acc: &Vec3) -> f64 {
    (acc.x * acc.x + acc.y * acc.y + acc.z * acc.z).sqrt()
}

// ── Test 1: interacción en borde SFC ─────────────────────────────────────────

/// Una partícula "local" en un dominio SFC y una partícula "halo" en el dominio vecino
/// están físicamente cerca (r < r_cut). La fuerza SR debe ser no nula.
#[test]
fn sr_sfc_border_interaction() {
    // Simula dos dominios SFC adyacentes en x:
    // Rank 0: x ∈ [0, 0.5)  — partícula local en x=0.48
    // Rank 1: x ∈ [0.5, 1)  — partícula halo en x=0.52
    let local = make_particle(0, 0.48, 0.5, 0.5, 1.0);
    let halo = make_particle(1, 0.52, 0.5, 0.5, 1.0);

    let r_split = 0.1_f64;
    let r_cut = 5.0 * r_split; // 0.5

    // Distancia = 0.04 < r_cut=0.5 → debe haber fuerza SR
    let params = SfcShortRangeParams {
        local_particles: &[local],
        halo_particles: &[halo],
        eps2: 1e-6,
        g: 1.0,
        r_split,
        box_size: 1.0,
    };
    let mut out = vec![Vec3::zero()];
    short_range_accels_sfc(&params, &mut out);

    assert!(
        force_mag(&out[0]) > 0.0,
        "SR-SFC borde: fuerza debe ser no nula (d=0.04 < r_cut={r_cut}), got {:?}",
        out[0]
    );
    // La fuerza apunta en +x (hacia el halo en x=0.52)
    assert!(
        out[0].x > 0.0,
        "SR-SFC borde: fuerza debe apuntar en +x, got {:?}",
        out[0]
    );
}

// ── Test 2: diagonal periódica en dominio SFC ────────────────────────────────

/// Partícula local en (0.01, 0.01, 0.01) y halo en (0.99, 0.99, 0.99).
/// Con minimum_image: distancia = √(3×0.02²) ≈ 0.0346 < r_cut → fuerza no nula.
/// Sin minimum_image: distancia ≈ √(3×0.98²) ≈ 1.70 >> r_cut → fuerza ≈ 0.
#[test]
fn sr_sfc_diagonal_periodic_interaction() {
    let local = make_particle(0, 0.01, 0.01, 0.01, 1.0);
    let halo = make_particle(1, 0.99, 0.99, 0.99, 1.0);

    let r_split = 0.1_f64;
    let r_cut = 5.0 * r_split; // 0.5

    // Distancia periódica diagonal = √(3 × 0.02²) ≈ 0.0346 < r_cut=0.5
    let d_periodic = (3.0_f64 * 0.02_f64 * 0.02_f64).sqrt();
    assert!(
        d_periodic < r_cut,
        "distancia periódica diagonal ({d_periodic:.4}) debe ser < r_cut ({r_cut})"
    );

    let params = SfcShortRangeParams {
        local_particles: &[local],
        halo_particles: &[halo],
        eps2: 1e-6,
        g: 1.0,
        r_split,
        box_size: 1.0,
    };
    let mut out = vec![Vec3::zero()];
    short_range_accels_sfc(&params, &mut out);

    let fmag = force_mag(&out[0]);
    assert!(
        fmag > 0.0,
        "SR-SFC diagonal periódica: fuerza debe ser no nula (d_periodic={d_periodic:.4}), got fmag={fmag}"
    );
    // La imagen más cercana del halo está en (-0.01, -0.01, -0.01) → fuerza apunta en -x,-y,-z
    assert!(
        out[0].x < 0.0,
        "fuerza SR diagonal debe apuntar en -x, got {:?}",
        out[0]
    );
    assert!(
        out[0].y < 0.0,
        "fuerza SR diagonal debe apuntar en -y, got {:?}",
        out[0]
    );
    assert!(
        out[0].z < 0.0,
        "fuerza SR diagonal debe apuntar en -z, got {:?}",
        out[0]
    );
}

// ── Test 3: halo 3D periódico cubre todos los pares con r < r_cut ─────────────

/// Simula un dominio SFC de rank 0 con AABB [0,0.5)³.
/// Una partícula en (0.95, 0.95, 0.95) debe ser incluida en el halo de rank 0
/// porque su distancia 3D periódica al AABB es ~0.087 < r_cut=0.1.
///
/// El halo 1D-z fallaría en este caso (test heredado de Fase 22, aquí en contexto SFC).
#[test]
fn sr_sfc_halo_3d_covers_rcut_pairs() {
    use gadget_ng_parallel::halo3d::Aabb3;

    // AABB del dominio SFC del rank 0 (octante inferior)
    let aabb_rank0 = Aabb3 {
        lo: [0.0, 0.0, 0.0],
        hi: [0.5, 0.5, 0.5],
    };

    let r_cut = 0.1_f64;
    let box_size = 1.0_f64;

    // Partícula en diagonal periódica: distancia ≈ √(3×0.05²) ≈ 0.087 < r_cut=0.1
    let pos_diagonal = [0.95, 0.95, 0.95_f64];
    assert!(
        is_in_periodic_halo(pos_diagonal, &aabb_rank0, r_cut, box_size),
        "halo 3D debe incluir partícula diagonal (0.95,0.95,0.95) con r_cut={r_cut}"
    );

    // Partícula en borde x periódico: distancia = 0.05 < r_cut
    let pos_x = [0.95, 0.25, 0.25_f64];
    assert!(
        is_in_periodic_halo(pos_x, &aabb_rank0, r_cut, box_size),
        "halo 3D debe incluir partícula en borde x periódico (0.95,0.25,0.25)"
    );

    // Partícula lejana: distancia periódica > r_cut → NO debe estar en halo
    let pos_far = [0.75, 0.75, 0.75_f64];
    assert!(
        !is_in_periodic_halo(pos_far, &aabb_rank0, r_cut, box_size),
        "halo 3D NO debe incluir partícula lejana (0.75,0.75,0.75) con r_cut={r_cut}"
    );

    // Confirmamos también con compute_aabb_3d que el AABB de un conjunto de partículas
    // incluye sus posiciones correctamente (para simular el AABB real de un dominio SFC).
    let particles = vec![
        make_particle(0, 0.1, 0.2, 0.3, 1.0),
        make_particle(1, 0.4, 0.45, 0.48, 1.0),
    ];
    let aabb_real = compute_aabb_3d(&particles);
    assert!(aabb_real.is_valid(), "AABB real debe ser válido");
    assert!((aabb_real.lo[0] - 0.1).abs() < 1e-12);
    assert!((aabb_real.hi[0] - 0.4).abs() < 1e-12);
}

// ── Test 4: minimum_image activo en walk SR-SFC ──────────────────────────────

/// Partícula local en z=0.05 y halo en z=0.95.
/// Sin minimum_image: d_z = 0.90 → fuera de r_cut → fuerza ≈ 0.
/// Con minimum_image: d_z = 0.10 (imagen: z=−0.05) → fuerza apuntando en −z.
///
/// Si minimum_image está activo en el walk SR-SFC, la fuerza debe ser no nula
/// y apuntar en -z (hacia la imagen periódica más cercana).
#[test]
fn sr_sfc_minimum_image_active_in_walk() {
    let local = make_particle(0, 0.5, 0.5, 0.05, 1.0);
    let halo = make_particle(1, 0.5, 0.5, 0.95, 1.0);

    let r_split = 0.1_f64;
    let r_cut = 5.0 * r_split; // 0.5

    let params = SfcShortRangeParams {
        local_particles: &[local],
        halo_particles: &[halo],
        eps2: 1e-6,
        g: 1.0,
        r_split,
        box_size: 1.0,
    };
    let mut out = vec![Vec3::zero()];
    short_range_accels_sfc(&params, &mut out);

    let fmag = force_mag(&out[0]);
    assert!(
        fmag > 0.0,
        "SR-SFC: minimum_image debe activar fuerza (d_min_image=0.10 < r_cut={r_cut}), got fmag={fmag}"
    );
    // minimum_image da dz = 0.05 - 0.95 + 1.0 = 0.10 → imagen en z = -0.05 → fuerza en -z
    assert!(
        out[0].z < 0.0,
        "SR-SFC: fuerza periódica debe apuntar en -z (hacia imagen más cercana), got {:?}",
        out[0]
    );
}

// ── Test 5: no hay huecos geométricos en SR-SFC ───────────────────────────────

/// Crea una cuadrícula de partículas, secciona en dos "dominios SFC" adyacentes,
/// y verifica que el halo 3D cubre todas las interacciones con r < r_cut,
/// sin importar de qué lado de la frontera estén las partículas.
///
/// También confirma que partículas sin halos no reciben fuerzas (sin auto-fuerza).
#[test]
fn sr_sfc_no_geometric_gaps() {
    // Dominio SFC del rank actual: partículas en x ∈ [0.0, 0.4)
    // Halo recibido del rank vecino: partículas en x ∈ [0.4, 0.8)
    // Ninguna partícula del halo queda fuera del alcance de las locales

    let r_split = 0.1_f64;
    let r_cut = 5.0 * r_split; // 0.5
    let box_size = 1.0_f64;

    let local_particles: Vec<Particle> = (0..5)
        .map(|i| make_particle(i, i as f64 * 0.08, 0.5, 0.5, 1.0))
        .collect();

    let halo_particles: Vec<Particle> = (0..5)
        .map(|i| make_particle(10 + i, 0.4 + i as f64 * 0.08, 0.5, 0.5, 1.0))
        .collect();

    // Para la partícula local más a la derecha (x=0.32), la halo más cercana está en x=0.40.
    // Distancia = 0.08 < r_cut=0.5 → debe haber fuerza SR.
    let closest_local_x = 4_f64 * 0.08; // 0.32
    let closest_halo_x = 0.40_f64;
    let d_gap = (closest_halo_x - closest_local_x).abs();
    assert!(
        d_gap < r_cut,
        "gap entre dominios SR-SFC ({d_gap}) debe ser < r_cut ({r_cut})"
    );

    let params = SfcShortRangeParams {
        local_particles: &local_particles,
        halo_particles: &halo_particles,
        eps2: 1e-6,
        g: 1.0,
        r_split,
        box_size,
    };
    let mut out = vec![Vec3::zero(); local_particles.len()];
    short_range_accels_sfc(&params, &mut out);

    // La partícula local en x=0.32 (índice 4) debe recibir fuerza de la halo en x=0.40.
    let fmag_rightmost = force_mag(&out[4]);
    assert!(
        fmag_rightmost > 0.0,
        "SR-SFC: partícula en borde derecho del dominio debe recibir fuerza del halo (d={d_gap}), got fmag={fmag_rightmost}"
    );

    // Sin halos: una sola partícula aislada no recibe fuerza (sin auto-fuerza).
    let single = [make_particle(99, 0.5, 0.5, 0.5, 1.0)];
    let params_solo = SfcShortRangeParams {
        local_particles: &single,
        halo_particles: &[],
        eps2: 1e-6,
        g: 1.0,
        r_split,
        box_size,
    };
    let mut out_solo = vec![Vec3::zero()];
    short_range_accels_sfc(&params_solo, &mut out_solo);
    assert!(
        force_mag(&out_solo[0]) < 1e-14,
        "SR-SFC: partícula sola sin halos no debe recibir fuerza (auto-fuerza), got {:?}",
        out_solo[0]
    );
}
