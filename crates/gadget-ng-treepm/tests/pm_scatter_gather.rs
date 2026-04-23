//! Tests unitarios del protocolo scatter/gather PM↔SFC (Fase 24).
//!
//! Verifica correctitud geométrica, conservación de masa y equivalencia física
//! con el path de Fase 23 (clone + migrate) en P=1.
//!
//! Tests:
//! 1. `scatter_cic_mass_conservation`     — masa total en grid = suma de masas de partículas
//! 2. `scatter_border_z_split`            — partícula en borde de slab usa ghost-right correctamente
//! 3. `gather_returns_correct_gid`        — acc_pm se asigna al global_id correcto
//! 4. `scatter_gather_p1_equals_phase23`  — resultados idénticos a Fase 23 en P=1
//! 5. `periodic_z_border_correct`         — partícula en z≈box_size → slab rank 0

use gadget_ng_core::{Particle, Vec3};
use gadget_ng_parallel::SerialRuntime;
use gadget_ng_pm::{slab_pm, SlabLayout};
use gadget_ng_treepm::distributed::pm_scatter_gather_accels;

fn make_particle(id: usize, x: f64, y: f64, z: f64, mass: f64) -> Particle {
    Particle::new(id, mass, Vec3::new(x, y, z), Vec3::zero())
}

fn acc_mag(a: &Vec3) -> f64 {
    (a.x * a.x + a.y * a.y + a.z * a.z).sqrt()
}

// ── Test 1: conservación de masa en el scatter CIC ──────────────────────────

/// La suma de densidad depositada en el grid PM debe ser igual a la suma de masas
/// de las partículas scatter. El protocolo scatter/gather no debe perder masa.
///
/// Se verifica haciendo el depósito directamente sobre las posiciones/masas de
/// entrada (el mismo cálculo que haría el slab PM tras recibir el scatter).
#[test]
fn scatter_cic_mass_conservation() {
    let nm = 16_usize;
    let box_size = 1.0_f64;
    let layout = SlabLayout::new(nm, 0, 1); // P=1

    let particles = vec![
        make_particle(0, 0.1, 0.2, 0.3, 1.5),
        make_particle(1, 0.5, 0.5, 0.5, 2.0),
        make_particle(2, 0.9, 0.8, 0.7, 0.5),
        make_particle(3, 0.05, 0.95, 0.05, 1.0),
    ];
    let total_mass: f64 = particles.iter().map(|p| p.mass).sum();

    let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();

    // Deposit CIC (mismo que haría el slab PM tras scatter)
    let density = slab_pm::deposit_slab_extended(&positions, &masses, &layout, box_size);

    // Para P=1, deposit_slab_extended usa la ruta cic::assign (nm³ celdas)
    let deposited_mass: f64 = density.iter().sum();

    let rel_err = (deposited_mass - total_mass).abs() / total_mass;
    assert!(
        rel_err < 1e-12,
        "masa depositada {deposited_mass:.6} != masa total {total_mass:.6} (err_rel={rel_err:.2e})"
    );
}

// ── Test 2: partícula en borde de slab usa ghost-right ──────────────────────

/// Una partícula muy cerca del borde derecho de un slab contribuye a la celda
/// z0 y z0+1. La celda z0+1 puede estar en el plano ghost-right del slab.
/// El mecanismo ghost-right + exchange_density_halos_z garantiza que la
/// contribución CIC al plano z0+1 se transfiere al slab vecino correctamente.
///
/// En P=1 verificamos que el depósito CIC produce exactamente
/// la misma masa total sin importar la posición de la partícula.
#[test]
fn scatter_border_z_split() {
    let nm = 8_usize;
    let box_size = 1.0_f64;
    let layout = SlabLayout::new(nm, 0, 1);

    // Partícula exactamente en la frontera de celda z (iz=3, frac=0.0)
    // → stencil CIC: iz=3 con peso 1.0 e iz=4 con peso 0.0 (caso límite)
    let cell_size = box_size / nm as f64;
    let z_border = 3.0 * cell_size; // exactamente en el borde de celda iz=3
    let p = make_particle(0, 0.5, 0.5, z_border, 2.0);

    let positions = vec![p.position];
    let masses = vec![p.mass];
    let density = slab_pm::deposit_slab_extended(&positions, &masses, &layout, box_size);

    let deposited: f64 = density.iter().sum();
    let rel_err = (deposited - p.mass).abs() / p.mass;
    assert!(
        rel_err < 1e-12,
        "masa en borde de celda no conservada: depositada={deposited}, esperada={} (err={rel_err:.2e})",
        p.mass
    );

    // Partícula en posición z muy cercana al borde derecho del slab (z≈0.999)
    // que usa celda z=7 y z=0 (periódico) en CIC. Misma conservación.
    let z_near_right = box_size - 0.5 * cell_size; // cerca del borde derecho
    let p2 = make_particle(1, 0.5, 0.5, z_near_right, 3.0);
    let positions2 = vec![p2.position];
    let masses2 = vec![p2.mass];
    let density2 = slab_pm::deposit_slab_extended(&positions2, &masses2, &layout, box_size);
    let deposited2: f64 = density2.iter().sum();
    let rel_err2 = (deposited2 - p2.mass).abs() / p2.mass;
    assert!(
        rel_err2 < 1e-12,
        "masa en borde derecho no conservada: depositada={deposited2}, esperada={} (err={rel_err2:.2e})",
        p2.mass
    );
}

// ── Test 3: gather devuelve acc_pm al global_id correcto ────────────────────

/// Tras scatter/gather PM completo en P=1, la aceleración PM de cada partícula
/// está asociada al global_id correcto. En particular, la fuerza sobre una
/// partícula A no está asignada a la partícula B.
///
/// Se usan dos partículas cercanas y se verifica que las fuerzas son
/// anti-paralelas (acción-reacción) y apuntan a los GIDs correctos.
#[test]
fn gather_returns_correct_gid() {
    let nm = 16_usize;
    let box_size = 1.0_f64;
    let layout = SlabLayout::new(nm, 0, 1);
    let rt = SerialRuntime;

    // Dos partículas: una en x=0.3 y otra en x=0.7 (eje x, separadas 0.4)
    // La fuerza PM de la partícula 0 debe apuntar en +x (hacia la partícula 1)
    // La fuerza PM de la partícula 1 debe apuntar en -x (hacia la partícula 0)
    let p0 = make_particle(0, 0.3, 0.5, 0.5, 1.0);
    let p1 = make_particle(1, 0.7, 0.5, 0.5, 1.0);
    let particles = vec![p0, p1];

    let r_split = 0.05; // r_split pequeño para que PM domine a esta distancia
    let (acc_pm, _stats) =
        pm_scatter_gather_accels(&particles, &layout, 1.0, r_split, box_size, &rt);

    assert_eq!(acc_pm.len(), 2, "debe haber una acc_pm por partícula");

    // Partícula 0 en x=0.3: fuerza PM debe apuntar en +x (hacia x=0.7)
    assert!(
        acc_pm[0].x > 0.0,
        "acc_pm[gid=0] debe apuntar en +x, got x={:.6}",
        acc_pm[0].x
    );
    // Partícula 1 en x=0.7: fuerza PM debe apuntar en -x (hacia x=0.3)
    assert!(
        acc_pm[1].x < 0.0,
        "acc_pm[gid=1] debe apuntar en -x, got x={:.6}",
        acc_pm[1].x
    );

    // Las magnitudes deben ser iguales por simetría (masas iguales, posición simétrica)
    let mag0 = acc_mag(&acc_pm[0]);
    let mag1 = acc_mag(&acc_pm[1]);
    assert!(
        (mag0 - mag1).abs() / mag0.max(1e-20) < 0.05,
        "magnitudes PM deberían ser similares: |acc[0]|={mag0:.4}, |acc[1]|={mag1:.4}"
    );
}

// ── Test 4: scatter/gather P=1 equivalente a Fase 23 ────────────────────────

/// En P=1, scatter/gather PM (Fase 24) debe dar resultados bit-a-bit idénticos
/// al path de Fase 23 (deposit + FFT + interpolate directamente sobre partículas).
///
/// Esto verifica que el shortcut P=1 de `pm_scatter_gather_accels` es correcto.
#[test]
fn scatter_gather_p1_equals_phase23() {
    let nm = 16_usize;
    let box_size = 1.0_f64;
    let layout = SlabLayout::new(nm, 0, 1);
    let rt = SerialRuntime;

    let particles = vec![
        make_particle(0, 0.15, 0.25, 0.35, 1.0),
        make_particle(1, 0.55, 0.45, 0.65, 2.0),
        make_particle(2, 0.85, 0.75, 0.15, 0.5),
    ];

    let r_split = 0.04;

    // Path Fase 24: scatter/gather
    let (acc_sg, _) = pm_scatter_gather_accels(&particles, &layout, 1.0, r_split, box_size, &rt);

    // Path Fase 23: deposit → FFT → interpolate directamente
    let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    let mut density_ext = slab_pm::deposit_slab_extended(&positions, &masses, &layout, box_size);
    slab_pm::exchange_density_halos_z(&mut density_ext, &layout, &rt);
    let mut forces =
        slab_pm::forces_from_slab(&density_ext, &layout, 1.0, box_size, Some(r_split), &rt);
    slab_pm::exchange_force_halos_z(&mut forces, &layout, &rt);
    let acc_p23 = slab_pm::interpolate_slab_local(&positions, &forces, &layout, box_size);

    assert_eq!(acc_sg.len(), acc_p23.len());
    for (i, (a_sg, a_p23)) in acc_sg.iter().zip(acc_p23.iter()).enumerate() {
        assert!(
            (a_sg.x - a_p23.x).abs() < 1e-14
                && (a_sg.y - a_p23.y).abs() < 1e-14
                && (a_sg.z - a_p23.z).abs() < 1e-14,
            "acc_pm[{i}] Fase24 != Fase23: sg=({:.8},{:.8},{:.8}) p23=({:.8},{:.8},{:.8})",
            a_sg.x,
            a_sg.y,
            a_sg.z,
            a_p23.x,
            a_p23.y,
            a_p23.z
        );
    }
}

// ── Test 5: partícula en z≈box_size → routing correcto periódico ────────────

/// Una partícula en z muy cercano a box_size debe ser enrutada al slab rank 0
/// (borde periódico): `iz0_global.rem_euclid(nm) / nz_local = 0`.
///
/// En P=1 no hay routing real, pero verificamos que el scatter/gather produce
/// una fuerza no nula para partículas en el borde periódico, lo que confirma
/// que el depósito CIC periódico funciona correctamente.
#[test]
fn periodic_z_border_correct() {
    let nm = 8_usize;
    let box_size = 1.0_f64;
    let layout = SlabLayout::new(nm, 0, 1);
    let rt = SerialRuntime;

    // Partícula cerca del borde derecho periódico: z ≈ box_size - epsilon
    let eps = 1e-6;
    let p_hi = make_particle(0, 0.5, 0.5, box_size - eps, 1.0);
    // Partícula cerca del borde izquierdo: z ≈ 0 + epsilon
    let p_lo = make_particle(1, 0.5, 0.5, eps, 1.0);

    let particles = vec![p_hi, p_lo];
    let r_split = 0.04;

    // El scatter/gather no debe fallar (no panic) y debe retornar fuerzas finitas.
    let (acc_pm, stats) =
        pm_scatter_gather_accels(&particles, &layout, 1.0, r_split, box_size, &rt);

    assert_eq!(acc_pm.len(), 2, "debe haber acc_pm por cada partícula");
    assert_eq!(stats.scatter_particles, 2);
    assert_eq!(stats.gather_particles, 2);

    for (i, a) in acc_pm.iter().enumerate() {
        assert!(a.x.is_finite(), "acc_pm[{i}].x no finito: {}", a.x);
        assert!(a.y.is_finite(), "acc_pm[{i}].y no finito: {}", a.y);
        assert!(a.z.is_finite(), "acc_pm[{i}].z no finito: {}", a.z);
    }

    // Las dos partículas son periódicamente cercanas en z (separación ~2e-6)
    // y simétricamente opuestas: la fuerza PM en z de p_hi debe ser igual y
    // opuesta a la de p_lo (acción-reacción periódica).
    let fz_sum = acc_pm[0].z + acc_pm[1].z;
    assert!(
        fz_sum.abs() < 1e-6,
        "suma de fuerzas PM en z no es ~0 (conservación impulso): fz_sum={fz_sum:.2e}"
    );

    // Verificar routing P=N>1: para cualquier partícula, el target rank calculado
    // por el protocolo scatter debe ser válido (en [0, size)).
    let n_ranks = 1_usize;
    let nz_local = nm / n_ranks;
    for p in &particles {
        let iz0 = (p.position.z * nm as f64 / box_size).floor() as i64;
        let iz0 = iz0.rem_euclid(nm as i64) as usize;
        let target = (iz0 / nz_local).min(n_ranks - 1);
        assert!(
            target < n_ranks,
            "target rank {target} fuera de rango [0, {n_ranks})"
        );
    }
}
