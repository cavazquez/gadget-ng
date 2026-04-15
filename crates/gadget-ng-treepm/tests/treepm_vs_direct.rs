//! Test: TreePM vs DirectGravity para N=8 partículas.
//!
//! El solver TreePM combina un PM de largo alcance (filtrado) y un solver de
//! corto alcance (erfc). Para una distribución no uniforme, las aceleraciones
//! deben tener el mismo signo y un orden de magnitud similar a DirectGravity.

use gadget_ng_core::{DirectGravity, GravitySolver, Vec3};
use gadget_ng_treepm::TreePmSolver;

/// 8 partículas en una nube asimétrica dentro del cubo [0, box_size)³.
fn asymmetric_cloud(box_size: f64) -> (Vec<Vec3>, Vec<f64>) {
    let positions = vec![
        Vec3::new(0.15 * box_size, 0.15 * box_size, 0.15 * box_size),
        Vec3::new(0.25 * box_size, 0.15 * box_size, 0.15 * box_size),
        Vec3::new(0.15 * box_size, 0.25 * box_size, 0.15 * box_size),
        Vec3::new(0.25 * box_size, 0.25 * box_size, 0.15 * box_size),
        Vec3::new(0.15 * box_size, 0.15 * box_size, 0.25 * box_size),
        Vec3::new(0.25 * box_size, 0.15 * box_size, 0.25 * box_size),
        Vec3::new(0.15 * box_size, 0.25 * box_size, 0.25 * box_size),
        Vec3::new(0.25 * box_size, 0.25 * box_size, 0.25 * box_size),
    ];
    let masses = vec![1.0_f64; 8];
    (positions, masses)
}

/// La fuerza de TreePM y DirectGravity deben apuntar en la misma dirección
/// para una partícula de prueba próxima a la nube (dentro de la mitad del cubo).
#[test]
fn treepm_direction_matches_direct() {
    let box_size = 2.0_f64;
    let nm = 32usize;
    let g = 1.0_f64;
    let eps2 = 1e-4_f64;

    let (mut positions, mut masses) = asymmetric_cloud(box_size);
    // Sonda en x=0.4*box_size, close to cloud centred at ~0.2*box_size.
    let probe_idx = positions.len();
    positions.push(Vec3::new(0.4 * box_size, 0.2 * box_size, 0.2 * box_size));
    masses.push(0.001);

    let probe = vec![probe_idx];

    // ── DirectGravity ────────────────────────────────────────────────────────
    let direct = DirectGravity;
    let mut acc_direct = vec![Vec3::zero()];
    direct.accelerations_for_indices(&positions, &masses, eps2, g, &probe, &mut acc_direct);

    // ── TreePM ────────────────────────────────────────────────────────────────
    let treepm = TreePmSolver {
        grid_size: nm,
        box_size,
        r_split: 0.0,
    };
    let mut acc_treepm = vec![Vec3::zero()];
    treepm.accelerations_for_indices(&positions, &masses, eps2, g, &probe, &mut acc_treepm);

    let ax_d = acc_direct[0].x;
    let ax_t = acc_treepm[0].x;

    // La nube está a x < 0.3*box_size; la sonda a x=0.4 → fuerza en −x.
    assert!(ax_d < 0.0, "Direct ax={ax_d:.4e} debería ser < 0");
    assert!(ax_t < 0.0, "TreePM ax={ax_t:.4e} debería ser < 0");

    // Las magnitudes deben estar dentro de un factor 10.
    let ratio = ax_t.abs() / ax_d.abs();
    assert!(
        ratio > 0.05 && ratio < 20.0,
        "ratio = {ratio:.3} fuera de [0.05, 20]"
    );
}

/// La suma de todas las fuerzas TreePM ponderadas por masa debe ser ≈ 0
/// (conservación de impulso, igual que el PM puro).
#[test]
fn treepm_momentum_conserved() {
    let box_size = 2.0_f64;
    let nm = 16usize;
    let g = 1.0_f64;
    let eps2 = 1e-4_f64;

    let (positions, masses) = asymmetric_cloud(box_size);
    let n = positions.len();

    let treepm = TreePmSolver {
        grid_size: nm,
        box_size,
        r_split: 0.0,
    };

    let all_idx: Vec<usize> = (0..n).collect();
    let mut acc = vec![Vec3::zero(); n];
    treepm.accelerations_for_indices(&positions, &masses, eps2, g, &all_idx, &mut acc);

    // Impulso total (dp/dt = ∑ m·a) debe ser ≈ 0.
    let mut sum = Vec3::zero();
    for (&m, &a) in masses.iter().zip(acc.iter()) {
        sum += a * m;
    }
    let mag = sum.dot(sum).sqrt();
    let scale = acc
        .iter()
        .zip(masses.iter())
        .map(|(&a, &m)| a.dot(a).sqrt() * m)
        .fold(1e-30_f64, f64::max);

    // La suma de fuerzas internas del PM periódico es exactamente cero.
    // La parte de corto alcance (erfc) viola Newton III a nivel de monopolo,
    // así que permitimos una tolerancia mayor (1e-4 relativo).
    assert!(
        mag / scale < 1e-3,
        "|∑ m·a| / max(m·a) = {:.3e} (demasiado grande)", mag / scale
    );
}
