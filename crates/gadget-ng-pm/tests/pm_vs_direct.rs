//! Test: comparación cualitativa entre PM y DirectGravity.
//!
//! Para una distribución NO uniforme, el solver PM debe producir aceleraciones
//! con el mismo **signo** y **orden de magnitud** que `DirectGravity`, aunque
//! difieren cuantitativamente (PM suaviza a escala de celda).
//!
//! El test usa una nube asimétrica de 8 partículas: el centro de masa no está
//! en el origen, de modo que hay fuerzas netas no nulas en ambos solvers.

use gadget_ng_core::{DirectGravity, GravitySolver, Vec3};
use gadget_ng_pm::PmSolver;

/// Configuración simple: masa pesada en el centro, sonda a 0.1 box_size a la derecha.
/// La distancia directa (0.1) es mucho menor que la periódica (0.9), así que ambos
/// solvers deben coincidir en el signo de la fuerza (-x).
fn center_mass_probe(box_size: f64) -> (Vec<Vec3>, Vec<f64>, usize) {
    // Masa pesada en el centro del cubo.
    let mut positions = vec![Vec3::new(0.5 * box_size, 0.5 * box_size, 0.5 * box_size)];
    let mut masses = vec![10.0_f64];
    // Sonda a 0.1*box_size a la derecha del centro (distancia directa 0.1 << periódica 0.9).
    let probe_idx = 1;
    positions.push(Vec3::new(0.6 * box_size, 0.5 * box_size, 0.5 * box_size));
    masses.push(0.001); // masa insignificante para la sonda
    (positions, masses, probe_idx)
}

/// La aceleración de la sonda debe apuntar hacia la masa pesada en −x.
/// El PM y DirectGravity deben coincidir en el signo; la magnitud puede diferir
/// (el PM usa condiciones periódicas y suavizado a escala de celda).
#[test]
fn pm_force_direction_matches_direct() {
    let box_size = 1.0_f64;
    let nm = 32usize; // grid 32³ para resolución sub-celda ≈ 0.03
    let g = 1.0_f64;
    let eps2 = 1e-4_f64;

    let (positions, masses, probe_idx) = center_mass_probe(box_size);
    let probe_slice = vec![probe_idx];

    // ── DirectGravity (sin periodicidad) ─────────────────────────────────────
    let direct = DirectGravity;
    let mut acc_direct = vec![Vec3::zero(); 1];
    direct.accelerations_for_indices(&positions, &masses, eps2, g, &probe_slice, &mut acc_direct);

    // ── PM (periódico) ────────────────────────────────────────────────────────
    let pm = PmSolver {
        grid_size: nm,
        box_size,
    };
    let mut acc_pm = vec![Vec3::zero(); 1];
    pm.accelerations_for_indices(&positions, &masses, eps2, g, &probe_slice, &mut acc_pm);

    // La sonda está a +x del centro: la fuerza debe apuntar en −x (hacia la masa).
    let ax_direct = acc_direct[0].x;
    let ax_pm = acc_pm[0].x;

    assert!(
        ax_direct < 0.0,
        "DirectGravity: ax_direct={ax_direct:.4e} debería ser < 0"
    );
    assert!(
        ax_pm < 0.0,
        "PM: ax_pm={ax_pm:.4e} debería ser < 0 (misma dirección que DirectGravity)"
    );

    // Las magnitudes deben estar en el mismo orden de magnitud.
    // Factor 50× de margen: PM incluye imágenes periódicas (que aquí son débiles a 0.9 de distancia)
    // y suavizado de celda (Δx ≈ 0.03, distancia real 0.1 → corrección ~10 %).
    let ratio = ax_pm.abs() / ax_direct.abs();
    assert!(
        ratio > 0.05 && ratio < 50.0,
        "ratio |ax_pm/ax_direct| = {ratio:.3} fuera del rango [0.05, 50]"
    );
}

/// Para cualquier distribución la suma de fuerzas PM ponderada por masa debe ser ≈ 0.
/// La antisimetría del Green's function periódico garantiza la tercera ley de Newton
/// (el impulso total del sistema se conserva).
#[test]
fn pm_total_momentum_conserved() {
    let box_size = 1.0_f64;
    let nm = 16usize;
    let g = 1.0_f64;

    let (positions, masses, _) = center_mass_probe(box_size);
    let n = positions.len();

    let pm = PmSolver {
        grid_size: nm,
        box_size,
    };

    let all_idx: Vec<usize> = (0..n).collect();
    let mut acc = vec![Vec3::zero(); n];
    pm.accelerations_for_indices(&positions, &masses, 0.0, g, &all_idx, &mut acc);

    // Suma de fuerzas ponderadas por masa = dp/dt total. Para fuerzas internas puras,
    // la tercera ley de Newton garantiza que la suma es cero.
    // Con PM periódico hay fuerzas de imagen, pero la suma sigue siendo cero por la
    // antisimetría del Green's function G(k): sum_i m_i * a_i ≈ 0.
    let mut sum = Vec3::zero();
    for (&m, &a) in masses.iter().zip(acc.iter()) {
        sum += a * m;
    }
    let mag = sum.dot(sum).sqrt();
    let scale = acc
        .iter()
        .zip(masses.iter())
        .map(|(&a, &m)| a.dot(a).sqrt() * m)
        .fold(0.0_f64, f64::max);

    assert!(
        mag < 1e-8 * scale.max(1e-15),
        "suma de fuerzas no es nula: |∑ m·a| = {mag:.3e}, escala = {scale:.3e}"
    );
}
