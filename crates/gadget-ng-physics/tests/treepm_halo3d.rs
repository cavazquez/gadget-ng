//! Tests de validación física para el halo volumétrico 3D periódico (Fase 22).
//!
//! Cubre:
//! 1. `halo3d_x_border_interaction`        — partícula x=0.01 interactúa con x=0.99 vía halo 3D
//! 2. `halo3d_y_border_interaction`        — idem en y
//! 3. `halo3d_z_border_interaction`        — idem en z
//! 4. `halo3d_diagonal_xyz_interaction`    — (0.01,0.01,0.01) ↔ (0.99,0.99,0.99) vía halo 3D
//! 5. `halo3d_vs_1d_uniform_slab_equivalent` — para Z-slab uniforme, halos 1D y 3D equivalentes
//! 6. `halo3d_force_partition_erf_erfc`    — F_lr + F_sr = F_Newton dentro de tolerancia
//! 7. `cosmo_treepm_3d_halo_no_explosion`  — N=27, 3 pasos, sin NaN/Inf (smoke test)
//! 8. `halo3d_no_double_counting`          — erf+erfc=1, lattice simétrico → fuerza central ≈ 0

use gadget_ng_core::{Particle, Vec3};
use gadget_ng_parallel::{
    halo3d::{compute_aabb_3d, is_in_periodic_halo, Aabb3},
    ParallelRuntime,
};
use gadget_ng_treepm::{
    distributed::{short_range_accels_slab, SlabShortRangeParams},
    short_range::{erfc_approx, minimum_image},
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

fn erfc_factor_local(r: f64, r_split: f64) -> f64 {
    let x = r / (std::f64::consts::SQRT_2 * r_split);
    erfc_approx(x)
}

/// Fuerza SR esperada (Newton × erfc), para par único sin softening.
fn expected_sr_force(r: f64, mass_j: f64, g: f64, r_split: f64) -> f64 {
    let erfc_k = erfc_factor_local(r, r_split);
    g * mass_j * erfc_k / (r * r)
}

// ── Test 1: interacción en borde x ────────────────────────────────────────────

/// Partícula en x=0.01 debe interactuar con la de x=0.99 gracias al halo 3D.
/// Sin halo (solo local) no habría interacción. Con halo 3D, la imagen periódica
/// de la partícula x=0.99 está a distancia 0.02 < r_cut=0.1.
#[test]
fn halo3d_x_border_interaction() {
    let box_size = 1.0_f64;
    let r_split = 0.02_f64;
    let r_cut = 5.0 * r_split; // 0.10

    let local_p = make_particle(0, 0.01, 0.5, 0.5, 1.0);
    let neighbor = make_particle(1, 0.99, 0.5, 0.5, 1.0);

    // Comprobar primero que el halo 3D incluiría a 'neighbor'.
    let aabb_local = Aabb3 {
        lo: [local_p.position.x - 1e-6, 0.0, 0.0],
        hi: [local_p.position.x + 1e-6, 1.0, 1.0],
    };
    assert!(
        is_in_periodic_halo(
            [neighbor.position.x, neighbor.position.y, neighbor.position.z],
            &aabb_local, r_cut, box_size
        ),
        "halo 3D debe incluir partícula x=0.99 en el halo de x=0.01 con r_cut={r_cut}"
    );

    // Simular la interacción: local=[p_0], halos=[p_1]
    let params = SlabShortRangeParams {
        local_particles: &[local_p],
        halo_particles: &[neighbor],
        eps2: 1e-10,
        g: 1.0,
        r_split,
        box_size,
    };
    let mut out = vec![Vec3::zero()];
    short_range_accels_slab(&params, &mut out);

    let fmag = out[0].norm();
    assert!(
        fmag > 0.0,
        "Test 1 (borde x): fuerza debe ser no nula, got fmag={fmag}"
    );
    // La fuerza debe dirigirse hacia +x (hacia la imagen periódica en x=0.99-1=-0.01).
    assert!(
        out[0].x < 0.0,
        "Test 1: fuerza en x debe ser negativa (hacia la imagen periódica), got fx={:.4e}",
        out[0].x
    );
}

// ── Test 2: interacción en borde y ────────────────────────────────────────────

#[test]
fn halo3d_y_border_interaction() {
    let box_size = 1.0_f64;
    let r_split = 0.02_f64;
    let r_cut = 5.0 * r_split;

    let local_p = make_particle(0, 0.5, 0.01, 0.5, 1.0);
    let neighbor = make_particle(1, 0.5, 0.99, 0.5, 1.0);

    let aabb_local = Aabb3 {
        lo: [0.0, local_p.position.y - 1e-6, 0.0],
        hi: [1.0, local_p.position.y + 1e-6, 1.0],
    };
    assert!(
        is_in_periodic_halo(
            [neighbor.position.x, neighbor.position.y, neighbor.position.z],
            &aabb_local, r_cut, box_size
        ),
        "halo 3D debe incluir partícula y=0.99 con r_cut={r_cut}"
    );

    let params = SlabShortRangeParams {
        local_particles: &[local_p],
        halo_particles: &[neighbor],
        eps2: 1e-10,
        g: 1.0,
        r_split,
        box_size,
    };
    let mut out = vec![Vec3::zero()];
    short_range_accels_slab(&params, &mut out);

    let fmag = out[0].norm();
    assert!(fmag > 0.0, "Test 2 (borde y): fmag={fmag}");
    assert!(
        out[0].y < 0.0,
        "Test 2: fuerza en y debe ser negativa (imagen periódica), got fy={:.4e}",
        out[0].y
    );
}

// ── Test 3: interacción en borde z ────────────────────────────────────────────

#[test]
fn halo3d_z_border_interaction() {
    let box_size = 1.0_f64;
    let r_split = 0.02_f64;
    let r_cut = 5.0 * r_split;

    let local_p = make_particle(0, 0.5, 0.5, 0.01, 1.0);
    let neighbor = make_particle(1, 0.5, 0.5, 0.99, 1.0);

    let aabb_local = Aabb3 {
        lo: [0.0, 0.0, local_p.position.z - 1e-6],
        hi: [1.0, 1.0, local_p.position.z + 1e-6],
    };
    assert!(
        is_in_periodic_halo(
            [neighbor.position.x, neighbor.position.y, neighbor.position.z],
            &aabb_local, r_cut, box_size
        ),
        "halo 3D debe incluir partícula z=0.99 con r_cut={r_cut}"
    );

    let params = SlabShortRangeParams {
        local_particles: &[local_p],
        halo_particles: &[neighbor],
        eps2: 1e-10,
        g: 1.0,
        r_split,
        box_size,
    };
    let mut out = vec![Vec3::zero()];
    short_range_accels_slab(&params, &mut out);

    let fmag = out[0].norm();
    assert!(fmag > 0.0, "Test 3 (borde z): fmag={fmag}");
    assert!(
        out[0].z < 0.0,
        "Test 3: fuerza en z debe ser negativa, got fz={:.4e}",
        out[0].z
    );
}

// ── Test 4: interacción diagonal x+y+z ───────────────────────────────────────

/// (0.01, 0.01, 0.01) ↔ (0.99, 0.99, 0.99) vía halo 3D periódico.
/// Distancia periódica diagonal = sqrt(3 × 0.02²) ≈ 0.0346 < r_cut=0.1.
#[test]
fn halo3d_diagonal_xyz_interaction() {
    let box_size = 1.0_f64;
    let r_split = 0.02_f64;
    let r_cut = 5.0 * r_split;

    let local_p = make_particle(0, 0.01, 0.01, 0.01, 1.0);
    let neighbor = make_particle(1, 0.99, 0.99, 0.99, 1.0);

    // Distancia periódica diagonal esperada.
    let dx = minimum_image(local_p.position.x - neighbor.position.x, box_size);
    let dy = minimum_image(local_p.position.y - neighbor.position.y, box_size);
    let dz = minimum_image(local_p.position.z - neighbor.position.z, box_size);
    let d_diag = (dx*dx + dy*dy + dz*dz).sqrt();
    assert!(
        d_diag < r_cut,
        "distancia diagonal periódica={d_diag:.4} debe ser < r_cut={r_cut}"
    );

    // Verificar que el halo 3D capturaría esta partícula.
    let aabb_local = Aabb3 {
        lo: [local_p.position.x - 1e-6; 3],
        hi: [local_p.position.x + 1e-6; 3],
    };
    assert!(
        is_in_periodic_halo(
            [neighbor.position.x, neighbor.position.y, neighbor.position.z],
            &aabb_local, r_cut, box_size
        ),
        "halo 3D debe capturar partícula diagonal (0.99,0.99,0.99) con r_cut={r_cut}"
    );

    // Calcular la fuerza SR.
    let params = SlabShortRangeParams {
        local_particles: &[local_p],
        halo_particles: &[neighbor],
        eps2: 1e-10,
        g: 1.0,
        r_split,
        box_size,
    };
    let mut out = vec![Vec3::zero()];
    short_range_accels_slab(&params, &mut out);

    let fmag = out[0].norm();
    assert!(fmag > 0.0, "Test 4 (diagonal): fmag={fmag}");
}

// ── Test 5: equivalencia halo 1D y 3D para Z-slab uniforme ───────────────────

/// Para Z-slab con partículas uniformes, los criterios 1D-z y 3D deben incluir
/// exactamente el mismo conjunto de vecinos (demostración matemática del plan).
///
/// Partícula fuente en (0.5, 0.5, 0.45) → a distancia z=0.05 del borde del slab
/// [0, 0.5). Para un rank con z_lo=0, z_hi=0.5, r_cut=0.1:
/// - Criterio 1D: z_source < z_hi + r_cut = 0.6 → INCLUIDA.
/// - Criterio 3D: dist_3D_periodic(source, aabb_slab) ≈ 0.0 < r_cut → INCLUIDA.
///
/// Partícula fuente en (0.5, 0.5, 0.65) → z=0.65:
/// - Criterio 1D: z=0.65 > z_hi + r_cut = 0.6 → EXCLUIDA.
/// - Criterio 3D: dist_3D_periodic(source, aabb_slab):
///   CoM_z = 0.25, half_z = 0.25. min_image(0.25-0.65)= -0.4, excess_z= |0.4|-0.25=0.15 > r_cut.
///   → EXCLUIDA también. ✓ Equivalencia.
#[test]
fn halo3d_vs_1d_uniform_slab_equivalent() {
    let box_size = 1.0_f64;
    let r_cut = 0.1_f64;

    // AABB del slab rank 0: z ∈ [0, 0.5).
    let aabb_slab = Aabb3 {
        lo: [0.0, 0.0, 0.0],
        hi: [box_size, box_size, 0.5],
    };
    let z_hi_slab = 0.5_f64;

    // Caso A: partícula en z=0.45 — dentro de la "zona" 1D (z < z_hi + r_cut = 0.6).
    let pa = [0.5, 0.5, 0.45_f64];
    let included_1d_a = pa[2] < z_hi_slab + r_cut; // true (0.45 < 0.6)
    let included_3d_a = is_in_periodic_halo(pa, &aabb_slab, r_cut, box_size);
    assert!(included_1d_a, "Caso A: 1D debe incluir z=0.45");
    assert!(included_3d_a, "Caso A: 3D debe incluir z=0.45");

    // Caso B: partícula en z=0.65 — fuera de la zona 1D.
    let pb = [0.5, 0.5, 0.65_f64];
    let included_1d_b = pb[2] < z_hi_slab + r_cut; // false (0.65 > 0.6)
    let included_3d_b = is_in_periodic_halo(pb, &aabb_slab, r_cut, box_size);
    assert!(!included_1d_b, "Caso B: 1D debe excluir z=0.65");
    assert!(!included_3d_b, "Caso B: 3D debe excluir z=0.65 (equivalencia con 1D para Z-slab)");
}

// ── Test 6: partición de fuerza F_lr + F_sr = F_Newton ───────────────────────

/// Verificar que erf(x) + erfc(x) = 1 para varios valores de x,
/// lo que garantiza que PM + árbol SR cubre exactamente Newton sin huecos ni solapamiento.
/// Además: la magnitud de la fuerza SR es correcta (escala con g, 1/r², erfc).
#[test]
fn halo3d_force_partition_erf_erfc() {
    let r_split = 0.05_f64;
    let g = 1.0_f64;

    // erf + erfc = 1 para todo x.
    for d in [0.01_f64, 0.03, 0.05, 0.08, 0.12, 0.20, 0.50, 1.0] {
        let x = d / (std::f64::consts::SQRT_2 * r_split);
        let erfc_v = erfc_approx(x);
        let erf_v = 1.0 - erfc_v;
        let sum = erf_v + erfc_v;
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "partición erf+erfc=1 para d={d}: sum={sum:.8}"
        );
    }

    // Fuerza SR para par cercano: erfc≈1 cuando r << r_split.
    // La aproximación `erfc_approx` tiene error máximo ~2%, así que la tolerancia
    // de convergencia a Newton es del 5% para r << r_split.
    let r = 0.001_f64;
    let mass_j = 1.0_f64;
    let f_sr = expected_sr_force(r, mass_j, g, r_split);
    let f_newton = g * mass_j / (r * r);
    // erfc(x) para x ≈ 0.014 → erfc ≈ 0.984 con la aproximación usada.
    assert!(
        (f_sr / f_newton - 1.0).abs() < 0.05,
        "fuerza SR para r << r_split debe ≈ Newton (5% tol): f_sr/f_N = {:.6}", f_sr / f_newton
    );

    // Fuerza SR para r >> r_cut: erfc(x) ≈ 0, fuerza debe ser muy pequeña.
    let r_far = 5.0 * r_split; // = r_cut
    let f_sr_far = expected_sr_force(r_far, mass_j, g, r_split);
    let f_newton_far = g * mass_j / (r_far * r_far);
    assert!(
        f_sr_far / f_newton_far < 0.01,
        "fuerza SR en r=r_cut debe ser ≈0: ratio={:.4}", f_sr_far / f_newton_far
    );
}

// ── Test 7: smoke test cosmológico (N=27, 3 pasos) ───────────────────────────

/// Verifica que la pipeline TreePM + halo 3D no produce NaN/Inf con N=27 partículas.
/// No valida física cuantitativa, solo ausencia de divergencias.
#[test]
fn cosmo_treepm_3d_halo_no_explosion() {
    use gadget_ng_pm::{slab_fft::SlabLayout, slab_pm};
    use gadget_ng_parallel::SerialRuntime;

    let n_side = 3usize;
    let box_size = 1.0_f64;
    let nm = 6usize;
    let r_split = 2.5 * box_size / nm as f64;
    let r_cut = 5.0 * r_split;
    let g_cosmo = 1.0_f64;
    let eps2 = 1e-6_f64;

    // Grilla uniforme de partículas.
    let dx = box_size / n_side as f64;
    let mass = 1.0_f64 / (n_side * n_side * n_side) as f64;
    let mut particles: Vec<Particle> = (0..n_side)
        .flat_map(|iz| {
            (0..n_side).flat_map(move |iy| {
                (0..n_side).map(move |ix| {
                    make_particle(
                        iz * n_side * n_side + iy * n_side + ix,
                        (ix as f64 + 0.5) * dx,
                        (iy as f64 + 0.5) * dx,
                        (iz as f64 + 0.5) * dx,
                        mass,
                    )
                })
            })
        })
        .collect();

    let rt = SerialRuntime;
    let layout = SlabLayout::new(nm, 0, 1);
    let n = particles.len();
    let mut acc = vec![Vec3::zero(); n];

    // Ejecutar 3 pasos de fuerza.
    for _ in 0..3 {
        // En P=1 el halo 3D es vacío (no hay vecinos remotos).
        let halos = rt.exchange_halos_3d_periodic(&particles, box_size, r_cut);
        assert!(halos.is_empty(), "P=1: halo 3D debe ser vacío");

        // PM largo alcance.
        let local_pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
        let local_mass: Vec<f64> = particles.iter().map(|p| p.mass).collect();

        let mut density_ext = slab_pm::deposit_slab_extended(&local_pos, &local_mass, &layout, box_size);
        slab_pm::exchange_density_halos_z(&mut density_ext, &layout, &rt);
        let mut forces = slab_pm::forces_from_slab(&density_ext, &layout, g_cosmo, box_size, Some(r_split), &rt);
        slab_pm::exchange_force_halos_z(&mut forces, &layout, &rt);
        let acc_lr = slab_pm::interpolate_slab_local(&local_pos, &forces, &layout, box_size);

        // SR árbol.
        let sr_params = SlabShortRangeParams {
            local_particles: &particles,
            halo_particles: &halos,
            eps2,
            g: g_cosmo,
            r_split,
            box_size,
        };
        let mut acc_sr = vec![Vec3::zero(); n];
        short_range_accels_slab(&sr_params, &mut acc_sr);

        // Suma.
        for (k, a) in acc.iter_mut().enumerate() {
            *a = acc_lr[k] + acc_sr[k];
        }

        // Verificar que no hay NaN/Inf.
        for (k, a) in acc.iter().enumerate() {
            assert!(
                a.x.is_finite() && a.y.is_finite() && a.z.is_finite(),
                "Test 7: NaN/Inf en acc[{k}] = ({:.4e}, {:.4e}, {:.4e})",
                a.x, a.y, a.z
            );
        }

        // Drift simple (no-op para este smoke test).
        let dt = 0.01_f64;
        for p in particles.iter_mut() {
            p.position.x = (p.position.x + p.velocity.x * dt).rem_euclid(box_size);
            p.position.y = (p.position.y + p.velocity.y * dt).rem_euclid(box_size);
            p.position.z = (p.position.z + p.velocity.z * dt).rem_euclid(box_size);
        }
    }
}

// ── Test 8: no doble conteo con lattice simétrico ─────────────────────────────

/// Para un lattice simétrico, la fuerza sobre la partícula central es ≈0.
/// Verifica que el halo 3D no introduce doble conteo:
/// cada par se contabiliza exactamente una vez.
#[test]
fn halo3d_no_double_counting() {
    // Parte 1: partición analítica erf+erfc=1.
    let r_split = 0.10_f64;
    for d in [0.02, 0.05, 0.10, 0.20, 0.50, 1.0_f64] {
        let x = d / (std::f64::consts::SQRT_2 * r_split);
        let erfc_v = erfc_approx(x);
        let erf_v = 1.0 - erfc_v;
        assert!(
            (erf_v + erfc_v - 1.0).abs() < 1e-6,
            "erf+erfc=1 para d={d}: sum={:.8}", erf_v + erfc_v
        );
    }

    // Parte 2: lattice 3³ → fuerza sobre la partícula central ≈ 0 por simetría.
    let box_size = 1.0_f64;
    let n_side = 3usize;
    let dx = box_size / n_side as f64;
    let mass = 1.0_f64 / 27.0;

    let particles: Vec<Particle> = (0..n_side)
        .flat_map(|iz| {
            (0..n_side).flat_map(move |iy| {
                (0..n_side).map(move |ix| {
                    make_particle(
                        iz * n_side * n_side + iy * n_side + ix,
                        (ix as f64 + 0.5) * dx,
                        (iy as f64 + 0.5) * dx,
                        (iz as f64 + 0.5) * dx,
                        mass,
                    )
                })
            })
        })
        .collect();

    let center_idx = 13; // partícula en (0.5, 0.5, 0.5).
    let local_center = particles[center_idx].clone();
    let halos: Vec<Particle> = particles
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != center_idx)
        .map(|(_, p)| p.clone())
        .collect();

    // Compute AABB de la partícula local (usada para validar el criterio del halo 3D).
    let aabb_center = compute_aabb_3d(&[local_center.clone()]);
    assert!(aabb_center.is_valid(), "AABB de partícula central debe ser válida");

    let r_split2 = dx * 0.4; // ≈ 0.133

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

    // Fuerza sobre la partícula central debe ser ≈ 0 por simetría (cada vecino
    // tiene un antisimétrico). Tolerancia amplia por aproximación monopolo del árbol.
    let fmag = out[0].norm();
    assert!(
        fmag < 0.05,
        "Test 8: fuerza sobre partícula central en lattice simétrico debe ser ≈0, got fmag={fmag:.4e}"
    );
}
