//! Test: distribución uniforme de masa → fuerzas PM ≈ 0.
//!
//! Una distribución perfectamente uniforme no tiene perturbaciones de densidad,
//! por lo que todas las componentes de Fourier de `δρ = ρ - ρ̄` son cero (el modo
//! DC se anula por construcción). El solver PM debe devolver aceleraciones nulas.

use gadget_ng_core::{GravitySolver, Vec3};
use gadget_ng_pm::PmSolver;

/// Red cúbica perfectamente uniforme: NP³ partículas igualmente espaciadas.
fn uniform_lattice(np: usize, box_size: f64) -> (Vec<Vec3>, Vec<f64>) {
    let step = box_size / np as f64;
    let mut positions = Vec::with_capacity(np * np * np);
    for iz in 0..np {
        for iy in 0..np {
            for ix in 0..np {
                positions.push(Vec3::new(
                    (ix as f64 + 0.5) * step,
                    (iy as f64 + 0.5) * step,
                    (iz as f64 + 0.5) * step,
                ));
            }
        }
    }
    let n = positions.len();
    let masses = vec![1.0_f64; n];
    (positions, masses)
}

/// Fuerzas en una red uniforme deben ser ≈ 0.
#[test]
fn uniform_lattice_gives_zero_acceleration() {
    let np = 4usize; // 4³ = 64 partículas
    let nm = 16usize; // grid 16³
    let box_size = 1.0_f64;

    let (positions, masses) = uniform_lattice(np, box_size);
    let n = positions.len();

    let solver = PmSolver {
        grid_size: nm,
        box_size,
    };

    let all_idx: Vec<usize> = (0..n).collect();
    let mut acc = vec![Vec3::zero(); n];

    solver.accelerations_for_indices(&positions, &masses, 0.0, 1.0, &all_idx, &mut acc);

    let max_acc = acc.iter().map(|a| a.dot(*a).sqrt()).fold(0.0_f64, f64::max);

    assert!(
        max_acc < 1e-10,
        "aceleración máxima en red uniforme = {max_acc:.3e} (esperado ≈ 0)"
    );
}

/// Sólo los índices solicitados reciben aceleración; los demás no se calculan.
#[test]
fn partial_indices_subset() {
    let np = 4usize;
    let nm = 8usize;
    let box_size = 1.0_f64;

    let (positions, masses) = uniform_lattice(np, box_size);
    let n = positions.len();
    let solver = PmSolver {
        grid_size: nm,
        box_size,
    };

    // Solo calcular para las primeras 4 partículas.
    let subset: Vec<usize> = (0..4).collect();
    let mut acc = vec![Vec3::zero(); 4];
    solver.accelerations_for_indices(&positions, &masses, 0.0, 1.0, &subset, &mut acc);

    // Red uniforme → aceleraciones ≈ 0 para el subconjunto también.
    let max_acc = acc.iter().map(|a| a.dot(*a).sqrt()).fold(0.0_f64, f64::max);
    assert!(max_acc < 1e-10, "acc max subset = {max_acc:.3e}");

    // El vector `acc` tiene exactamente 4 elementos (no n).
    assert_eq!(acc.len(), 4);

    // No hay acceso fuera de bounds; si llega hasta aquí el test pasa.
    let _ = n;
}
