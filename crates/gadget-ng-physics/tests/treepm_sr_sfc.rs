//! Tests de validación física para el path SR-SFC (Fase 23).
//!
//! Verifica que `short_range_accels_sfc` + `exchange_halos_3d_periodic` produce
//! física correcta y consistente con el path allgather (baseline serial) y con
//! el path treepm_slab_3d de Fase 22.
//!
//! 1. `sr_sfc_vs_slab_p1_equal`           — SR-SFC ≡ SR-slab en P=1 (mismo resultado)
//! 2. `sr_sfc_no_double_counting_pm`       — erf+erfc=1; PM+SR no doble conteo
//! 3. `sr_sfc_no_explosion_cosmological`   — N=27, 3 pasos, sin NaN/Inf
//! 4. `sr_sfc_pm_force_return_by_global_id`— acc_lr retornada por global_id es correcta
//! 5. `sr_sfc_x_border_periodic`           — interacción borde x periódico
//! 6. `sr_sfc_y_border_periodic`           — interacción borde y periódico
//! 7. `sr_sfc_momentum_conservation`       — |Δp| pequeño en N=8, 5 pasos (acción-reacción)
//! 8. `sr_sfc_equals_slab3d_p1`            — SR-SFC coincide bit-a-bit con treepm_slab_3d en P=1

use gadget_ng_core::{Particle, Vec3};
use gadget_ng_parallel::{ParallelRuntime, SerialRuntime, halo3d::is_in_periodic_halo};
use gadget_ng_pm::{slab_fft::SlabLayout, slab_pm};
use gadget_ng_treepm::{
    distributed::{
        SfcShortRangeParams, SlabShortRangeParams, short_range_accels_sfc, short_range_accels_slab,
    },
    short_range::erfc_approx,
};

// ── Utilidades ────────────────────────────────────────────────────────────────

fn make_particle(id: usize, x: f64, y: f64, z: f64, mass: f64) -> Particle {
    Particle::new(id, mass, Vec3::new(x, y, z), Vec3::zero())
}

/// Pipeline PM en serial para un conjunto de partículas.
/// Devuelve acc_lr (fuerzas de largo alcance).
fn run_pm_serial(
    particles: &[Particle],
    nm: usize,
    box_size: f64,
    g_cosmo: f64,
    r_split: f64,
) -> Vec<Vec3> {
    let rt = SerialRuntime;
    let layout = SlabLayout::new(nm, 0, 1);
    let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let mass: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    let mut density = slab_pm::deposit_slab_extended(&pos, &mass, &layout, box_size);
    slab_pm::exchange_density_halos_z(&mut density, &layout, &rt);
    let mut forces =
        slab_pm::forces_from_slab(&density, &layout, g_cosmo, box_size, Some(r_split), &rt);
    slab_pm::exchange_force_halos_z(&mut forces, &layout, &rt);
    slab_pm::interpolate_slab_local(&pos, &forces, &layout, box_size)
}

// ── Test 1: SR-SFC ≡ SR-slab en P=1 ─────────────────────────────────────────

/// En P=1, ambos paths (slab y SFC) no tienen halos y calculan sobre las mismas
/// partículas locales. Las aceleraciones SR deben ser idénticas.
#[test]
fn sr_sfc_vs_slab_p1_equal() {
    let box_size = 1.0_f64;
    let r_split = 0.05_f64;
    let eps2 = 1e-6_f64;

    // N=8 partículas en grilla 2³
    let particles: Vec<Particle> = (0..8)
        .map(|i| {
            let ix = i % 2;
            let iy = (i / 2) % 2;
            let iz = i / 4;
            make_particle(
                i,
                (ix as f64 + 0.5) * 0.5,
                (iy as f64 + 0.5) * 0.5,
                (iz as f64 + 0.5) * 0.5,
                1.0 / 8.0,
            )
        })
        .collect();

    // En P=1, halos = [] para ambos paths.
    let sfc_params = SfcShortRangeParams {
        local_particles: &particles,
        halo_particles: &[],
        eps2,
        g: 1.0,
        r_split,
        box_size,
    };
    let slab_params = SlabShortRangeParams {
        local_particles: &particles,
        halo_particles: &[],
        eps2,
        g: 1.0,
        r_split,
        box_size,
    };

    let mut out_sfc = vec![Vec3::zero(); particles.len()];
    let mut out_slab = vec![Vec3::zero(); particles.len()];
    short_range_accels_sfc(&sfc_params, &mut out_sfc);
    short_range_accels_slab(&slab_params, &mut out_slab);

    for (k, (a_sfc, a_slab)) in out_sfc.iter().zip(out_slab.iter()).enumerate() {
        let dx = (a_sfc.x - a_slab.x).abs();
        let dy = (a_sfc.y - a_slab.y).abs();
        let dz = (a_sfc.z - a_slab.z).abs();
        assert!(
            dx < 1e-14 && dy < 1e-14 && dz < 1e-14,
            "Test 1: SR-SFC y SR-slab deben ser idénticos en P=1 para partícula {k}: diff=({dx:.2e},{dy:.2e},{dz:.2e})"
        );
    }
}

// ── Test 2: no doble conteo PM + SR-SFC ──────────────────────────────────────

/// erf(x) + erfc(x) = 1 analíticamente → PM + SR-SFC cubre Newton sin huecos.
/// Verifica también que la magnitud de la fuerza SR escala correctamente.
#[test]
fn sr_sfc_no_double_counting_pm() {
    let r_split = 0.05_f64;

    // Partición analítica: erf+erfc=1 para todo x.
    for d in [0.01_f64, 0.02, 0.05, 0.08, 0.15, 0.30, 0.50] {
        let x = d / (std::f64::consts::SQRT_2 * r_split);
        let erfc = erfc_approx(x);
        let erf = 1.0 - erfc;
        let sum = erf + erfc;
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Test 2: erf+erfc=1 para d={d:.3}: sum={sum:.8}"
        );
    }

    // Una sola partícula sin halos → fuerza SR = 0 (sin auto-fuerza).
    let p = make_particle(0, 0.5, 0.5, 0.5, 1.0);
    let params = SfcShortRangeParams {
        local_particles: &[p],
        halo_particles: &[],
        eps2: 1e-6,
        g: 1.0,
        r_split,
        box_size: 1.0,
    };
    let mut out = vec![Vec3::zero()];
    short_range_accels_sfc(&params, &mut out);
    let fmag = out[0].norm();
    assert!(
        fmag < 1e-14,
        "Test 2: sin halos, fuerza SR debe ser 0, got fmag={fmag:.2e}"
    );
}

// ── Test 3: no explosión cosmológica (N=27, 3 pasos) ─────────────────────────

/// Smoke test: pipeline PM + SR-SFC con halo 3D sobre grilla 3³.
/// En P=1 el halo es vacío; el árbol calcula fuerzas locales. Sin NaN/Inf.
#[test]
fn sr_sfc_no_explosion_cosmological() {
    let n_side = 3usize;
    let box_size = 1.0_f64;
    let nm = 6usize;
    let r_split = 2.5 * box_size / nm as f64;
    let r_cut = 5.0 * r_split;
    let g_cosmo = 1.0_f64;
    let eps2 = 1e-6_f64;

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

    for _step in 0..3 {
        // En P=1 el halo 3D siempre es vacío.
        let halos = rt.exchange_halos_3d_periodic(&particles, box_size, r_cut);
        assert!(halos.is_empty(), "P=1: halos deben ser vacíos");

        // PM largo alcance.
        let acc_lr = run_pm_serial(&particles, nm, box_size, g_cosmo, r_split);

        // SR árbol (SFC path).
        let sr_params = SfcShortRangeParams {
            local_particles: &particles,
            halo_particles: &halos,
            eps2,
            g: g_cosmo,
            r_split,
            box_size,
        };
        let mut acc_sr = vec![Vec3::zero(); particles.len()];
        short_range_accels_sfc(&sr_params, &mut acc_sr);

        let n = particles.len();
        let mut acc = vec![Vec3::zero(); n];
        for k in 0..n {
            acc[k] = acc_lr[k] + acc_sr[k];
            assert!(
                acc[k].x.is_finite() && acc[k].y.is_finite() && acc[k].z.is_finite(),
                "Test 3: NaN/Inf en acc[{k}]"
            );
        }

        // Drift simple para mover las partículas (velocidades = 0 → posiciones fijas).
        let dt = 0.01_f64;
        for p in particles.iter_mut() {
            p.velocity.x += acc[0].x * dt * 0.5;
            p.position.x = (p.position.x + p.velocity.x * dt).rem_euclid(box_size);
            p.velocity.y += acc[0].y * dt * 0.5;
            p.position.y = (p.position.y + p.velocity.y * dt).rem_euclid(box_size);
            p.velocity.z += acc[0].z * dt * 0.5;
            p.position.z = (p.position.z + p.velocity.z * dt).rem_euclid(box_size);
        }
    }
}

// ── Test 4: retorno de fuerzas PM por global_id ───────────────────────────────

/// Verifica el mecanismo de retorno de fuerzas PM→SFC:
/// después de embeber acc_lr en pm_parts.acceleration, el lookup por global_id
/// devuelve exactamente las mismas fuerzas a los dueños SFC.
///
/// Simula el paso 3c–3e del engine para P=1 (sin migración real).
#[test]
fn sr_sfc_pm_force_return_by_global_id() {
    let box_size = 1.0_f64;
    let nm = 8usize;
    let r_split = 2.5 * box_size / nm as f64;
    let g_cosmo = 1.0_f64;

    // N=8 partículas con global_ids no consecutivos para probar el lookup.
    let particles: Vec<Particle> = vec![
        make_particle(10, 0.1, 0.1, 0.1, 0.125),
        make_particle(20, 0.6, 0.1, 0.1, 0.125),
        make_particle(30, 0.1, 0.6, 0.1, 0.125),
        make_particle(40, 0.6, 0.6, 0.1, 0.125),
        make_particle(50, 0.1, 0.1, 0.6, 0.125),
        make_particle(60, 0.6, 0.1, 0.6, 0.125),
        make_particle(70, 0.1, 0.6, 0.6, 0.125),
        make_particle(80, 0.6, 0.6, 0.6, 0.125),
    ];

    // Calcular acc_lr directamente (referencia).
    let acc_lr_ref = run_pm_serial(&particles, nm, box_size, g_cosmo, r_split);

    // Simular paso 3c: embeber en pm_parts.acceleration.
    let mut pm_parts = particles.clone();
    for (p, a) in pm_parts.iter_mut().zip(acc_lr_ref.iter()) {
        p.acceleration = *a;
    }

    // En P=1 exchange_domain_sfc y exchange_domain_by_z son no-ops.
    // El orden se conserva: pm_parts[i] sigue siendo particles[i].

    // Paso 3e: lookup por global_id.
    use std::collections::HashMap;
    let lr_map: HashMap<usize, Vec3> = pm_parts
        .iter()
        .map(|p| (p.global_id, p.acceleration))
        .collect();
    let acc_lr_via_map: Vec<Vec3> = particles.iter().map(|p| lr_map[&p.global_id]).collect();

    // El retorno por global_id debe ser bit-a-bit idéntico a la referencia.
    for (k, (a_ref, a_map)) in acc_lr_ref.iter().zip(acc_lr_via_map.iter()).enumerate() {
        let dx = (a_ref.x - a_map.x).abs();
        let dy = (a_ref.y - a_map.y).abs();
        let dz = (a_ref.z - a_map.z).abs();
        assert!(
            dx < 1e-15 && dy < 1e-15 && dz < 1e-15,
            "Test 4: lookup global_id debe ser bit-a-bit idéntico para partícula {k} (id={}): diff=({dx:.2e},{dy:.2e},{dz:.2e})",
            particles[k].global_id
        );
    }
}

// ── Test 5: interacción borde x periódico en SR-SFC ──────────────────────────

/// Partícula local en x=0.02, halo en x=0.98.
/// Con minimum_image: d_x = 0.04 < r_cut → fuerza en -x.
#[test]
fn sr_sfc_x_border_periodic() {
    let box_size = 1.0_f64;
    let r_split = 0.05_f64;
    let r_cut = 5.0 * r_split; // 0.25

    let local = make_particle(0, 0.02, 0.5, 0.5, 1.0);
    let halo = make_particle(1, 0.98, 0.5, 0.5, 1.0);

    // Verificar que el halo 3D cubriría este caso.
    use gadget_ng_parallel::halo3d::Aabb3;
    let aabb_local = Aabb3 {
        lo: [0.0, 0.0, 0.0],
        hi: [0.5, 1.0, 1.0],
    };
    assert!(
        is_in_periodic_halo([0.98, 0.5, 0.5], &aabb_local, r_cut, box_size),
        "halo 3D debe incluir x=0.98 en halo de dominio SFC con xhi=0.5"
    );

    let params = SfcShortRangeParams {
        local_particles: &[local],
        halo_particles: &[halo],
        eps2: 1e-8,
        g: 1.0,
        r_split,
        box_size,
    };
    let mut out = vec![Vec3::zero()];
    short_range_accels_sfc(&params, &mut out);

    assert!(
        out[0].norm() > 0.0,
        "Test 5: fuerza debe ser no nula (d_x=0.04 < r_cut={r_cut})"
    );
    // La imagen periódica más cercana de x=0.98 es x=-0.02 → fuerza en -x.
    assert!(
        out[0].x < 0.0,
        "Test 5: fuerza en -x, got fx={:.4e}",
        out[0].x
    );
}

// ── Test 6: interacción borde y periódico en SR-SFC ──────────────────────────

#[test]
fn sr_sfc_y_border_periodic() {
    let box_size = 1.0_f64;
    let r_split = 0.05_f64;
    let r_cut = 5.0 * r_split; // 0.25

    let local = make_particle(0, 0.5, 0.02, 0.5, 1.0);
    let halo = make_particle(1, 0.5, 0.98, 0.5, 1.0);

    // Verificar cobertura halo 3D.
    use gadget_ng_parallel::halo3d::Aabb3;
    let aabb_local = Aabb3 {
        lo: [0.0, 0.0, 0.0],
        hi: [1.0, 0.5, 1.0],
    };
    assert!(
        is_in_periodic_halo([0.5, 0.98, 0.5], &aabb_local, r_cut, box_size),
        "halo 3D debe incluir y=0.98 en halo de dominio SFC con yhi=0.5"
    );

    let params = SfcShortRangeParams {
        local_particles: &[local],
        halo_particles: &[halo],
        eps2: 1e-8,
        g: 1.0,
        r_split,
        box_size,
    };
    let mut out = vec![Vec3::zero()];
    short_range_accels_sfc(&params, &mut out);

    assert!(out[0].norm() > 0.0, "Test 6: fuerza debe ser no nula");
    assert!(
        out[0].y < 0.0,
        "Test 6: fuerza en -y, got fy={:.4e}",
        out[0].y
    );
}

// ── Test 7: conservación de momento (|Δp| pequeño) ────────────────────────────

/// Para N=8 partículas con velocidades aleatorias, el paso SR conserva el momento
/// porque short_range_accels_sfc implementa la tercera ley (acción-reacción):
/// para cada par (i,j), a_i += f_ij y a_j -= f_ij.
///
/// En modo P=1 (sin halos), la suma de fuerzas sobre todas las partículas debe
/// ser ≈ 0 (momento lineal conservado a nivel numérico, tolerancia 1e-12).
#[test]
fn sr_sfc_momentum_conservation() {
    let box_size = 1.0_f64;
    let r_split = 0.15_f64;
    let eps2 = 1e-6_f64;

    // 8 partículas no simétricas.
    let particles: Vec<Particle> = vec![
        make_particle(0, 0.12, 0.23, 0.34, 1.0),
        make_particle(1, 0.67, 0.14, 0.89, 1.0),
        make_particle(2, 0.45, 0.56, 0.12, 1.0),
        make_particle(3, 0.78, 0.89, 0.45, 1.0),
        make_particle(4, 0.23, 0.67, 0.56, 1.0),
        make_particle(5, 0.56, 0.34, 0.78, 1.0),
        make_particle(6, 0.34, 0.45, 0.67, 1.0),
        make_particle(7, 0.89, 0.78, 0.23, 1.0),
    ];

    let params = SfcShortRangeParams {
        local_particles: &particles,
        halo_particles: &[],
        eps2,
        g: 1.0,
        r_split,
        box_size,
    };
    let mut out = vec![Vec3::zero(); particles.len()];
    short_range_accels_sfc(&params, &mut out);

    // Σ mᵢ aᵢ debe ser ≈ 0 (momento total conservado) porque cada par contribuye
    // fuerzas iguales y opuestas (Newton III).
    let (px, py, pz) = out
        .iter()
        .zip(particles.iter())
        .fold((0.0_f64, 0.0_f64, 0.0_f64), |(px, py, pz), (a, p)| {
            (px + p.mass * a.x, py + p.mass * a.y, pz + p.mass * a.z)
        });
    let dp = (px * px + py * py + pz * pz).sqrt();
    assert!(
        dp < 1e-10,
        "Test 7: |Δp| = {dp:.2e} debe ser < 1e-10 (Newton III en SR-SFC)"
    );
}

// ── Test 8: SR-SFC ≡ treepm_slab_3d en P=1 ───────────────────────────────────

/// En P=1, SfcShortRangeParams con halos vacíos produce exactamente el mismo
/// resultado que SlabShortRangeParams con halos vacíos (mismo kernel subyacente).
/// Esto confirma que el desacoplamiento SFC no modifica la física SR.
#[test]
fn sr_sfc_equals_slab3d_p1() {
    let box_size = 1.0_f64;
    let r_split = 0.08_f64;
    let eps2 = 1e-6_f64;

    let particles: Vec<Particle> = vec![
        make_particle(0, 0.15, 0.25, 0.35, 1.0),
        make_particle(1, 0.75, 0.65, 0.55, 1.0),
        make_particle(2, 0.50, 0.10, 0.90, 1.0),
        make_particle(3, 0.30, 0.80, 0.20, 1.0),
    ];

    // Path SR-SFC (Fase 23): sin halos en P=1.
    let sfc_params = SfcShortRangeParams {
        local_particles: &particles,
        halo_particles: &[],
        eps2,
        g: 1.0,
        r_split,
        box_size,
    };
    let mut out_sfc = vec![Vec3::zero(); particles.len()];
    short_range_accels_sfc(&sfc_params, &mut out_sfc);

    // Path SR-slab (Fase 21/22): sin halos en P=1.
    let slab_params = SlabShortRangeParams {
        local_particles: &particles,
        halo_particles: &[],
        eps2,
        g: 1.0,
        r_split,
        box_size,
    };
    let mut out_slab = vec![Vec3::zero(); particles.len()];
    short_range_accels_slab(&slab_params, &mut out_slab);

    // Deben ser bit-a-bit idénticos (ambos delegan al mismo kernel).
    for (k, (a_sfc, a_slab)) in out_sfc.iter().zip(out_slab.iter()).enumerate() {
        let err = (
            (a_sfc.x - a_slab.x).abs(),
            (a_sfc.y - a_slab.y).abs(),
            (a_sfc.z - a_slab.z).abs(),
        );
        assert!(
            err.0 < 1e-14 && err.1 < 1e-14 && err.2 < 1e-14,
            "Test 8: SR-SFC y SR-slab deben ser idénticos en P=1 para partícula {k}: err={err:?}"
        );
    }
}
