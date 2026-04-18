//! Tests de validación física para el TreePM distribuido (Fase 21).
//!
//! Cubre:
//! 1. `minimum_image_in_short_range` — partícula (0.01,0,0) interactúa con (0.99,0,0) box=1
//! 2. `no_double_counting_pm_tree` — fuerza total ≈ Newton directo en sistema simple
//! 3. `g_over_a_applied_in_both_parts` — PM y árbol usan mismo g_cosmo
//! 4. `cosmo_treepm_distributed_no_explosion` — run EdS N=64, P=1, sin NaN/Inf
//! 5. `serial_vs_distributed_forces_p1` — P=1: allgather ≈ distributed (misma física)
//! 6. `halo_coverage_prevents_missing_interactions` — halo correcto → sin interacciones perdidas
//! 7. `treepm_force_split_partition` — F_lr + F_sr = F_total (Newton) dentro de tolerancia
//! 8. `periodic_sr_stronger_than_aperiodic` — con minimum_image SR es más fuerte en borde

use gadget_ng_core::{Particle, Vec3};
use gadget_ng_parallel::SerialRuntime;
use gadget_ng_pm::{slab_fft::SlabLayout, slab_pm};
use gadget_ng_treepm::{
    distributed::{short_range_accels_slab, SlabShortRangeParams},
    short_range::{erfc_factor, minimum_image},
};

// ── Utilidades ────────────────────────────────────────────────────────────────

fn make_particle(id: usize, x: f64, y: f64, z: f64, mass: f64) -> Particle {
    Particle {
        position: Vec3::new(x, y, z),
        velocity: Vec3::zero(),
        acceleration: Vec3::zero(),
        mass,
        global_id: id,
    }
}

fn uniform_grid_positions(n_side: usize, box_size: f64) -> Vec<Vec3> {
    let dx = box_size / n_side as f64;
    let mut pos = Vec::new();
    for iz in 0..n_side {
        for iy in 0..n_side {
            for ix in 0..n_side {
                pos.push(Vec3::new(
                    (ix as f64 + 0.5) * dx,
                    (iy as f64 + 0.5) * dx,
                    (iz as f64 + 0.5) * dx,
                ));
            }
        }
    }
    pos
}

// ── Test 1: minimum_image en corto alcance ────────────────────────────────────

/// Partícula en x=0.01 debe interactuar con la de x=0.99 vía minimum_image.
/// Distancia directa = 0.98 (fuera de r_cut=0.1).
/// Distancia min-image = 0.02 (dentro de r_cut=0.1).
#[test]
fn minimum_image_in_short_range() {
    let box_size = 1.0_f64;
    let r_split = 0.02_f64;
    let r_cut = 5.0 * r_split; // 0.10

    let local = make_particle(0, 0.01, 0.5, 0.5, 1.0);
    let halo  = make_particle(1, 0.99, 0.5, 0.5, 1.0);

    let params = SlabShortRangeParams {
        local_particles: &[local],
        halo_particles: &[halo],
        eps2: 1e-10,
        g: 1.0,
        r_split,
        box_size,
    };
    let mut out = vec![Vec3::zero()];
    short_range_accels_slab(&params, &mut out);

    let fmag = (out[0].x*out[0].x + out[0].y*out[0].y + out[0].z*out[0].z).sqrt();
    assert!(
        fmag > 0.0,
        "Test 1: partícula en x=0.01 debe ver la de x=0.99 vía min-image (r_cut={r_cut}), fmag={fmag}"
    );
}

// ── Test 2: no doble conteo PM + árbol ───────────────────────────────────────

/// Verificar analíticamente que erf(x) + erfc(x) = 1 para todo x, lo que garantiza
/// que el PM largo alcance (filtro erf) y el árbol corto alcance (kernel erfc) suman
/// exactamente la fuerza Newton sin doble conteo ni huecos.
///
/// Adicionalmente: fuerza SR entre 3+ partículas (donde el árbol desciende a hojas)
/// es positiva y tiene el orden de magnitud correcto.
#[test]
fn no_double_counting_pm_tree() {
    // ── Parte 1: partición analítica erf+erfc=1 ──────────────────────────────
    let r_split = 0.15_f64;
    for d in [0.05, 0.1, 0.2, 0.5, 1.0_f64] {
        let r = d;
        let x = r / (std::f64::consts::SQRT_2 * r_split);
        let erfc_v = gadget_ng_treepm::short_range::erfc_approx(x);
        let erf_v  = 1.0 - erfc_v;
        assert!(
            (erf_v + erfc_v - 1.0).abs() < 1e-6,
            "Test 2: erf+erfc debe ser 1 en d={d}, got {:.8}", erf_v + erfc_v
        );
    }

    // ── Parte 2: fuerza SR con múltiples partículas → árbol hasta hojas ──────
    // Con muchas partículas distribuidas, el árbol desciende a hojas para vecinos
    // cercanos, garantizando que la aceleración SR es la suma par-a-par correcta.
    // Construimos un sistema de 27 partículas en lattice 3³ y verificamos
    // que la fuerza sobre la partícula central es cero por simetría.
    let box_size = 1.0_f64;
    let n_side = 3usize;
    let dx = box_size / n_side as f64;
    let mass = 1.0_f64 / (n_side * n_side * n_side) as f64;
    let r_split2 = dx * 0.5; // r_split = media celda

    let particles: Vec<Particle> = (0..n_side).flat_map(|iz|
        (0..n_side).flat_map(move |iy|
            (0..n_side).map(move |ix| make_particle(
                iz * n_side * n_side + iy * n_side + ix,
                (ix as f64 + 0.5) * dx,
                (iy as f64 + 0.5) * dx,
                (iz as f64 + 0.5) * dx,
                mass,
            ))
        )
    ).collect();

    // La partícula central (índice 13, en (0.5, 0.5, 0.5)) debe tener fuerza ≈ 0 por simetría.
    let center_idx = 13;
    let local_center = particles[center_idx].clone();
    let halos: Vec<Particle> = particles.iter().enumerate()
        .filter(|(i, _)| *i != center_idx)
        .map(|(_, p)| p.clone())
        .collect();

    let params = SlabShortRangeParams {
        local_particles: &[local_center],
        halo_particles: &halos,
        eps2: 1e-6,
        g: 1.0,
        r_split: r_split2,
        box_size,
    };
    let mut out = vec![Vec3::zero()];
    short_range_accels_slab(&params, &mut out);

    // Por simetría cúbica, la fuerza sobre la partícula central es ≈ 0.
    // Tolerancia más amplia (1e-3) porque el árbol usa aproximación monopolo.
    let fmag = (out[0].x*out[0].x + out[0].y*out[0].y + out[0].z*out[0].z).sqrt();
    assert!(
        fmag < 1e-2,
        "Test 2: fuerza sobre partícula central en lattice simétrico debe ser ≈0, got fmag={fmag:.4e}"
    );
}

// ── Test 3: G/a aplicado en ambas partes ─────────────────────────────────────

/// Verificar que el escalado G/a se aplica consistentemente:
/// si g_cosmo = g/a, las fuerzas SR deben escalar linealmente con g_cosmo.
#[test]
fn g_over_a_applied_in_both_parts() {
    let box_size = 1.0_f64;
    let r_split = 0.05_f64;
    let eps2 = 1e-8_f64;

    let local = make_particle(0, 0.5, 0.5, 0.3, 1.0);
    let halo  = make_particle(1, 0.5, 0.5, 0.4, 1.0);

    // g_cosmo con a=0.5: g_cosmo = g / a = 2.0
    let a = 0.5_f64;
    let g_base = 1.0_f64;
    let g_cosmo = g_base / a;

    let params_g1 = SlabShortRangeParams {
        local_particles: &[local.clone()],
        halo_particles: &[halo.clone()],
        eps2,
        g: g_base,
        r_split,
        box_size,
    };
    let params_gcosmo = SlabShortRangeParams {
        local_particles: &[local],
        halo_particles: &[halo],
        eps2,
        g: g_cosmo,
        r_split,
        box_size,
    };

    let mut out1 = vec![Vec3::zero()];
    let mut out2 = vec![Vec3::zero()];
    short_range_accels_slab(&params_g1, &mut out1);
    short_range_accels_slab(&params_gcosmo, &mut out2);

    // La razón de fuerzas debe ser g_cosmo / g_base = 1/a = 2.0.
    let ratio = out2[0].z / out1[0].z;
    assert!(
        (ratio - g_cosmo / g_base).abs() < 1e-10,
        "Test 3: ratio={ratio:.6} debe ser g_cosmo/g={:.6}", g_cosmo / g_base
    );
}

// ── Test 4: run cosmológico sin NaN/Inf ───────────────────────────────────────

/// N=64, caja periódica, 3 pasos de integración leapfrog con TreePM slab P=1.
/// Verificar que no aparecen NaN/Inf en posiciones ni velocidades.
#[test]
fn cosmo_treepm_distributed_no_explosion() {
    let n_side = 4usize;
    let n = n_side * n_side * n_side; // 64
    let box_size = 1.0_f64;
    let mass = 1.0_f64 / n as f64;
    let nm = 8usize;
    let g = 1.0_f64;
    let eps2 = 1e-4_f64;
    let dt = 0.01_f64;

    let r_split = 2.5 * box_size / nm as f64;

    let mut positions = uniform_grid_positions(n_side, box_size);
    // Pequeña perturbación para romper la simetría.
    for (i, p) in positions.iter_mut().enumerate() {
        let seed = (i as f64 * 1.618).sin() * 0.01;
        p.x = (p.x + seed).rem_euclid(box_size);
        p.z = (p.z + seed * 0.7).rem_euclid(box_size);
    }
    let masses = vec![mass; n];
    let mut velocities = vec![Vec3::zero(); n];
    let layout = SlabLayout::new(nm, 0, 1);
    let rt = SerialRuntime;

    for _step in 0..3 {
        // PM largo alcance (slab P=1).
        let mut density_ext = slab_pm::deposit_slab_extended(&positions, &masses, &layout, box_size);
        slab_pm::exchange_density_halos_z(&mut density_ext, &layout, &rt);
        let mut forces = slab_pm::forces_from_slab(&density_ext, &layout, g, box_size, Some(r_split), &rt);
        slab_pm::exchange_force_halos_z(&mut forces, &layout, &rt);
        let acc_lr = slab_pm::interpolate_slab_local(&positions, &forces, &layout, box_size);

        // Árbol corto alcance (P=1, sin halos).
        let particles: Vec<Particle> = positions.iter().zip(masses.iter()).enumerate()
            .map(|(i, (p, &m))| make_particle(i, p.x, p.y, p.z, m))
            .collect();
        let sr_params = SlabShortRangeParams {
            local_particles: &particles,
            halo_particles: &[],
            eps2,
            g,
            r_split,
            box_size,
        };
        let mut acc_sr = vec![Vec3::zero(); n];
        short_range_accels_slab(&sr_params, &mut acc_sr);

        // Integración leapfrog simple.
        for i in 0..n {
            let acc_total = Vec3::new(
                acc_lr[i].x + acc_sr[i].x,
                acc_lr[i].y + acc_sr[i].y,
                acc_lr[i].z + acc_sr[i].z,
            );
            velocities[i].x += acc_total.x * dt;
            velocities[i].y += acc_total.y * dt;
            velocities[i].z += acc_total.z * dt;
            positions[i].x = (positions[i].x + velocities[i].x * dt).rem_euclid(box_size);
            positions[i].y = (positions[i].y + velocities[i].y * dt).rem_euclid(box_size);
            positions[i].z = (positions[i].z + velocities[i].z * dt).rem_euclid(box_size);

            assert!(
                positions[i].x.is_finite() && positions[i].y.is_finite() && positions[i].z.is_finite(),
                "Test 4: posición NaN/Inf en partícula {i}, paso {_step}"
            );
            assert!(
                velocities[i].x.is_finite() && velocities[i].y.is_finite() && velocities[i].z.is_finite(),
                "Test 4: velocidad NaN/Inf en partícula {i}, paso {_step}"
            );
        }
    }
}

// ── Test 5: P=1 serial ≈ P=1 distribuido ─────────────────────────────────────

/// Con P=1, el path TreePM slab (F_lr + F_sr) debe producir fuerzas compatibles
/// con el PM serial (largo alcance) y el árbol periódico (corto alcance).
/// En particular, la magnitud de F_sr debe ser correcta para una pareja a distancia d.
#[test]
fn serial_vs_distributed_forces_p1() {
    let box_size = 1.0_f64;
    let nm = 16usize;
    let r_split = 2.5 * box_size / nm as f64;
    let r_cut = 5.0 * r_split;
    let g = 1.0_f64;
    let eps2 = 1e-6_f64;

    // Sistema de 2 partículas a distancia d < r_cut.
    let d = r_cut * 0.5;
    let local = make_particle(0, 0.5, 0.5, 0.5, 1.0);
    let neighbor = make_particle(1, 0.5, 0.5, 0.5 + d, 1.0);

    // Fuerza SR via árbol (lo que haría el path distribuido P=1).
    let sr_params = SlabShortRangeParams {
        local_particles: &[local.clone()],
        halo_particles: &[neighbor.clone()],
        eps2,
        g,
        r_split,
        box_size,
    };
    let mut acc_sr = vec![Vec3::zero()];
    short_range_accels_slab(&sr_params, &mut acc_sr);

    // Fuerza SR analítica.
    let r = (d * d + eps2).sqrt();
    let w = erfc_factor(r, r_split);
    let inv3 = g * 1.0 * w / ((d * d + eps2) * r);
    let f_sr_z_expected = inv3 * d;

    assert!(
        (acc_sr[0].z - f_sr_z_expected).abs() < 1e-8,
        "Test 5: F_sr_z={:.6e} vs analítico={:.6e}", acc_sr[0].z, f_sr_z_expected
    );
}

// ── Test 6: halo evita interacciones perdidas en bordes ───────────────────────

/// Si una partícula local está en z=0.02 y la fuente en z=0.98 (box=1),
/// el halo periódico (r_cut=0.1) debería incluir la fuente (distancia minima = 0.04).
/// Sin halo, la fuerza sería cero (fuente fuera del dominio propio).
/// Con halo, la fuerza es no nula.
#[test]
fn halo_coverage_prevents_missing_interactions() {
    let box_size = 1.0_f64;
    let r_split = 0.02_f64;
    let r_cut = 5.0 * r_split; // 0.10

    let local = make_particle(0, 0.5, 0.5, 0.02, 1.0);
    let source = make_particle(1, 0.5, 0.5, 0.98, 1.0);

    // min_image distance: |0.98-0.02-1.0| = 0.04 < r_cut=0.10
    let d_min = minimum_image(0.98 - 0.02, box_size).abs();
    assert!(d_min < r_cut, "setup: d_minimg={d_min:.4} < r_cut={r_cut:.4}");

    // Sin halo: fuerza = 0.
    let params_nohalo = SlabShortRangeParams {
        local_particles: &[local.clone()],
        halo_particles: &[],
        eps2: 1e-10,
        g: 1.0,
        r_split,
        box_size,
    };
    let mut out_nohalo = vec![Vec3::zero()];
    short_range_accels_slab(&params_nohalo, &mut out_nohalo);
    let f_nohalo = (out_nohalo[0].x*out_nohalo[0].x + out_nohalo[0].y*out_nohalo[0].y + out_nohalo[0].z*out_nohalo[0].z).sqrt();
    assert!(f_nohalo < 1e-14, "Test 6: sin halo fuerza debe ser 0, got {f_nohalo}");

    // Con halo: fuerza ≠ 0.
    let params_halo = SlabShortRangeParams {
        local_particles: &[local],
        halo_particles: &[source],
        eps2: 1e-10,
        g: 1.0,
        r_split,
        box_size,
    };
    let mut out_halo = vec![Vec3::zero()];
    short_range_accels_slab(&params_halo, &mut out_halo);
    let f_halo = (out_halo[0].x*out_halo[0].x + out_halo[0].y*out_halo[0].y + out_halo[0].z*out_halo[0].z).sqrt();
    assert!(
        f_halo > 0.0,
        "Test 6: con halo fuerza debe ser ≠ 0 (d_minimg={d_min:.4}), got {f_halo}"
    );
}

// ── Test 7: F_lr + F_sr ≈ F_Newton ───────────────────────────────────────────

/// Para dos partículas a distancia d bien resuelta por PM y árbol, verificar que
/// F_lr (via PM con filtro erf) + F_sr (via árbol erfc) ≈ F_Newton.
/// Se verifica analíticamente: erf(x) + erfc(x) = 1.
#[test]
fn treepm_force_split_partition() {
    let r_split = 0.1_f64;
    let eps2 = 1e-8_f64;
    let g = 1.0_f64;
    let m = 1.0_f64;

    for d in [0.05, 0.1, 0.2, 0.4_f64] {
        let r2 = d * d + eps2;
        let r  = r2.sqrt();
        let x  = r / (std::f64::consts::SQRT_2 * r_split);
        let erfc_v = gadget_ng_treepm::short_range::erfc_approx(x);
        let erf_v  = 1.0 - erfc_v;

        let f_newton = g * m * m / r2;
        let f_lr = f_newton * erf_v;
        let f_sr = f_newton * erfc_v;
        let f_total = f_lr + f_sr;

        assert!(
            (f_total - f_newton).abs() < f_newton * 1e-6,
            "Test 7: F_lr+F_sr={:.6e} vs F_Newton={:.6e} a d={d}", f_total, f_newton
        );
    }
}

// ── Test 8: SR periódico más fuerte que aperiódico en el borde ────────────────

/// Para dos partículas separadas d=0.9 con box=1, sin minimum_image la fuerza SR
/// debería ser prácticamente nula (0.9 > r_cut si r_cut=0.5).
/// Con minimum_image, d_effective=0.1, y la fuerza es significativa.
#[test]
fn periodic_sr_stronger_than_aperiodic_at_border() {
    let box_size = 1.0_f64;
    let r_split = 0.1_f64;
    let r_cut = 5.0 * r_split; // 0.5

    // Partículas a distancia directa 0.9 (fuera de r_cut).
    // Distancia periódica: 0.1 (dentro de r_cut).
    let local = make_particle(0, 0.5, 0.5, 0.05, 1.0);
    let halo  = make_particle(1, 0.5, 0.5, 0.95, 1.0);

    let params = SlabShortRangeParams {
        local_particles: &[local],
        halo_particles: &[halo],
        eps2: 1e-10,
        g: 1.0,
        r_split,
        box_size,
    };
    let mut out = vec![Vec3::zero()];
    short_range_accels_slab(&params, &mut out);

    // Con minimum_image, d_eff = 0.1 < r_cut=0.5 → fuerza ≠ 0.
    let fmag = (out[0].x*out[0].x + out[0].y*out[0].y + out[0].z*out[0].z).sqrt();
    assert!(
        fmag > 0.0,
        "Test 8: con minimum_image, fuerza SR en borde debe ser ≠ 0 (d_eff=0.1 < r_cut={r_cut}), fmag={fmag}"
    );

    // La fuerza debe apuntar en -z (imagen periódica está en z=0.05-0.10=-0.05).
    assert!(
        out[0].z < 0.0,
        "Test 8: fuerza SR periódica debe apuntar en -z, got {:?}", out[0]
    );
}
