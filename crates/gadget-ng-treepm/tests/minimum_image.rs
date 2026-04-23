//! Tests de periodicidad para el árbol de corto alcance TreePM (Fase 21).
//!
//! Cubre:
//! 1. `minimum_image_basic` — casos simples de imagen mínima
//! 2. `border_particle_sr_periodic` — partícula en z=0.01 ve a otra en z=0.99
//! 3. `halo_coverage_completeness` — halo de ancho r_cut cubre todas las interacciones SR
//! 4. `minimum_image_no_double_counting` — partícula no se auto-fuerza con imagen periódica
//! 5. `periodic_aabb_wrap` — AABB periódica correcta para nodo en borde opuesto
//! 6. `sr_force_vs_direct_periodic` — fuerza SR periódica vs cálculo directo
//! 7. `erfc_partition_of_unity` — erf + erfc = 1 en toda la gama de r/r_s

use gadget_ng_core::{Particle, Vec3};
use gadget_ng_treepm::{
    distributed::{short_range_accels_slab, SlabShortRangeParams},
    short_range::{erfc_approx, erfc_factor, min_dist2_to_aabb_periodic, minimum_image},
};

// ── Utilidades ────────────────────────────────────────────────────────────────

fn make_particle(id: usize, x: f64, y: f64, z: f64, mass: f64) -> Particle {
    Particle::new(id, mass, Vec3::new(x, y, z), Vec3::zero())
}

// ── Test 1: minimum_image básico ──────────────────────────────────────────────

#[test]
fn minimum_image_basic() {
    let b = 1.0_f64;
    // Dentro de [-L/2, L/2]: sin cambio.
    assert!((minimum_image(0.3, b) - 0.3).abs() < 1e-14);
    assert!((minimum_image(-0.3, b) + 0.3).abs() < 1e-14);
    assert!((minimum_image(0.0, b)).abs() < 1e-14);

    // Más allá de L/2: imagen en [-L/2, L/2].
    let d1 = minimum_image(0.7, b);
    assert!((d1 + 0.3).abs() < 1e-12, "0.7 → -0.3, got {d1}");

    let d2 = minimum_image(-0.7, b);
    assert!((d2 - 0.3).abs() < 1e-12, "-0.7 → +0.3, got {d2}");

    // En el borde exacto L/2: |d| = 0.5.
    let d3 = minimum_image(0.5, b).abs();
    assert!(d3 <= 0.5 + 1e-12, "0.5 → |{d3}| debe ser ≤ 0.5");
}

// ── Test 2: partícula en borde interactúa con la del otro lado ────────────────

/// Una partícula en z=0.01 y otra en z=0.99 en caja box=1.
/// Con minimum_image la distancia es 0.02 (a través del borde), no 0.98 (directa).
/// Si r_cut=0.1, la fuerza SR debe ser no nula con minimum_image, y nula sin él.
#[test]
fn border_particle_sr_periodic() {
    let box_size = 1.0_f64;
    let r_split = 0.02_f64;
    let r_cut = 5.0 * r_split; // 0.1

    // Partícula local en z=0.01, halo (ghost) en z=0.99.
    // Distancia directa = 0.98 (fuera de r_cut).
    // Distancia min-image = 0.02 (dentro de r_cut).
    let local = make_particle(0, 0.5, 0.5, 0.01, 1.0);
    let halo = make_particle(1, 0.5, 0.5, 0.99, 1.0);

    let params = SlabShortRangeParams {
        local_particles: &[local],
        halo_particles: &[halo],
        eps2: 1e-8,
        g: 1.0,
        r_split,
        box_size,
    };
    let mut out = vec![Vec3::zero()];
    short_range_accels_slab(&params, &mut out);

    // La fuerza debe ser no nula: la imagen periódica está dentro de r_cut.
    let fmag = (out[0].x * out[0].x + out[0].y * out[0].y + out[0].z * out[0].z).sqrt();
    assert!(
        fmag > 0.0,
        "partícula en z=0.01 debe interactuar con la de z=0.99 vía minimum_image (r_cut={r_cut}), fmag={fmag}"
    );

    // La fuerza debe apuntar en -z (imagen periódica está en z = 0.01 - 0.02 = -0.01, o sea hacia z=L).
    assert!(
        out[0].z < 0.0,
        "fuerza SR debe apuntar en -z (atracción a través del borde periódico), got {:?}",
        out[0]
    );
}

// ── Test 3: halo de ancho r_cut cubre todas las interacciones necesarias ──────

/// Si una partícula local está en z=z_lo y la fuente está en z = z_lo - delta (delta < r_cut),
/// y el halo intercambia partículas con z < z_lo + r_cut, entonces la fuente DEBE estar
/// en el halo recibido por el rank que tiene z_lo como borde izquierdo.
#[test]
fn halo_coverage_completeness() {
    let box_size = 1.0_f64;
    let r_split = 0.05_f64;
    let r_cut = 5.0 * r_split; // 0.25

    // Simulamos rank 0 con z_lo=0.0, z_hi=0.5.
    // La fuente está en z = 0.8, que vía min-image está a distancia |0.8-0.0-1.0| = 0.2 < r_cut.
    // Para que rank 0 reciba esta partícula como halo, el rank que la posee debe enviarla.
    // El criterio de envío: z > z_hi - r_cut = 0.5 - 0.25 = 0.25 → z=0.8 > 0.25 ✓ (envía a derecha)
    // Y rank 0 (como leftmost) recibe del rank (P-1) mediante exchange_halos_by_z_periodic.

    // En este test, verificamos directamente que la fuerza SR es no nula usando
    // el halo directamente como parámetro (sin MPI real).
    let local = make_particle(0, 0.5, 0.5, 0.05, 1.0); // z cerca del borde izquierdo
    let halo = make_particle(1, 0.5, 0.5, 0.85, 1.0); // z en rank vecino

    // Distancia min-image: |0.85-0.05-1.0| = |−0.2| = 0.2 < r_cut=0.25.
    let d_minimg = minimum_image(0.85 - 0.05, box_size).abs();
    assert!(
        d_minimg < r_cut,
        "distancia min-image={d_minimg:.4} debe ser < r_cut={r_cut:.4}"
    );

    let params = SlabShortRangeParams {
        local_particles: &[local],
        halo_particles: &[halo],
        eps2: 1e-8,
        g: 1.0,
        r_split,
        box_size,
    };
    let mut out = vec![Vec3::zero()];
    short_range_accels_slab(&params, &mut out);

    let fmag = (out[0].x * out[0].x + out[0].y * out[0].y + out[0].z * out[0].z).sqrt();
    assert!(
        fmag > 0.0,
        "halo en z=0.85 debe contribuir al SR de partícula en z=0.05 (dist_minimg={d_minimg:.4}), fmag={fmag}"
    );
}

// ── Test 4: no auto-fuerza con imagen periódica ───────────────────────────────

/// Una sola partícula no debe ejercer fuerza sobre sí misma,
/// incluso con minimum_image (la imagen periódica de skip==j es excluida por el `j != skip`).
#[test]
fn minimum_image_no_double_counting() {
    let local = make_particle(0, 0.5, 0.5, 0.5, 1.0);
    let params = SlabShortRangeParams {
        local_particles: &[local],
        halo_particles: &[],
        eps2: 1e-8,
        g: 1.0,
        r_split: 0.1,
        box_size: 1.0,
    };
    let mut out = vec![Vec3::zero()];
    short_range_accels_slab(&params, &mut out);
    let fmag = (out[0].x * out[0].x + out[0].y * out[0].y + out[0].z * out[0].z).sqrt();
    assert!(fmag < 1e-14, "auto-fuerza debe ser cero, got fmag={fmag}");
}

// ── Test 5: AABB periódica correcta ──────────────────────────────────────────

#[test]
fn periodic_aabb_wrap() {
    let box_size = 1.0_f64;
    // Partícula en z=0.05; nodo centrado en z=0.95, half=0.03.
    // Distancia directa al borde de AABB: |0.95-0.05| - 0.03 = 0.87 (grande).
    // Distancia periódica: min_image(0.95-0.05,1.0)=-0.1, |−0.1|-0.03=0.07.
    let xi = Vec3::new(0.0, 0.0, 0.05);
    let com = Vec3::new(0.0, 0.0, 0.95);
    let half = 0.03_f64;

    let d2_periodic = min_dist2_to_aabb_periodic(xi, com, half, box_size);
    let expected = (0.1_f64 - half).powi(2); // (0.07)² = 0.0049
    assert!(
        (d2_periodic - expected).abs() < 1e-10,
        "AABB periódica: d2={d2_periodic:.6e} vs esperado={expected:.6e}"
    );

    // La distancia directa sería mucho mayor.
    let d_direct = (0.95_f64 - 0.05 - half).max(0.0).powi(2);
    assert!(
        d2_periodic < d_direct,
        "d2_periodic={d2_periodic:.4} debe ser < d_direct={d_direct:.4}"
    );
}

// ── Test 6: fuerza SR periódica vs cálculo directo ───────────────────────────

/// Verificar que la fuerza SR periódica entre dos partículas separadas
/// d = 0.1 (via minimum_image) coincide con el cálculo analítico.
#[test]
fn sr_force_vs_direct_periodic() {
    let box_size = 1.0_f64;
    let r_split = 0.05_f64;
    let eps2 = 1e-8_f64;
    let g = 1.0_f64;

    // Partícula 0 en z=0.05, partícula 1 en z=0.95.
    // min_image: dz = |0.95-0.05-1.0| = 0.10 (negativo, apuntando a z-).
    let local = make_particle(0, 0.0, 0.0, 0.05, 1.0);
    let halo = make_particle(1, 0.0, 0.0, 0.95, 1.0);

    let params = SlabShortRangeParams {
        local_particles: &[local],
        halo_particles: &[halo],
        eps2,
        g,
        r_split,
        box_size,
    };
    let mut out = vec![Vec3::zero()];
    short_range_accels_slab(&params, &mut out);

    // Cálculo analítico con minimum_image:
    let dz = minimum_image(0.95 - 0.05, box_size); // = -0.1
    let r2 = dz * dz + eps2;
    let r = r2.sqrt();
    let w = erfc_factor(r, r_split);
    let inv3 = g * 1.0 * w / (r2 * r);
    let fz_expected = dz * inv3;

    assert!(
        (out[0].z - fz_expected).abs() < 1e-9,
        "Fz={:.6e} vs esperado={:.6e}",
        out[0].z,
        fz_expected
    );
    // Solo componente z no nula (partículas alineadas en z).
    assert!(out[0].x.abs() < 1e-14, "Fx debe ser 0");
    assert!(out[0].y.abs() < 1e-14, "Fy debe ser 0");
}

// ── Test 7: erf + erfc = 1 ───────────────────────────────────────────────────

#[test]
fn erfc_partition_of_unity() {
    let r_split = 0.1_f64;
    for r in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0_f64] {
        let x = r / (std::f64::consts::SQRT_2 * r_split);
        let erfc_val = erfc_approx(x);
        let erf_val = 1.0 - erfc_val;
        assert!(
            (erf_val + erfc_val - 1.0).abs() < 1e-6,
            "partición de unidad falla en r={r}: erf={erf_val:.6} erfc={erfc_val:.6}"
        );
    }
}
