//! Tests de smoke para HipPmSolver.
//!
//! # Skip automático
//!
//! Los tests se marcan como `#[ignore]` cuando HIP no está disponible en tiempo de
//! ejecución (o `HIP_SKIP=1` en tiempo de compilación). Para ejecutarlos:
//!
//! ```bash
//! cargo test -p gadget-ng-hip --test hip_pm_smoke -- --ignored
//! ```
//!
//! # Estructura de los tests
//!
//! Idéntica a `gadget-ng-cuda/tests/cuda_pm_smoke.rs`, adaptada para HipPmSolver.

use gadget_ng_core::gravity::GravitySolver;
use gadget_ng_core::vec3::Vec3;
use gadget_ng_hip::HipPmSolver;

// ── Generador de partículas de prueba ─────────────────────────────────────────

fn test_particles(n: usize, box_size: f64) -> (Vec<Vec3>, Vec<f64>) {
    let positions = (0..n)
        .map(|i| {
            let t = i as f64 * 0.37;
            Vec3::new(
                (t.sin() * 0.5 + 0.5) * box_size,
                (t.cos() * 0.5 + 0.5) * box_size,
                ((t * 0.7).sin() * 0.5 + 0.5) * box_size,
            )
        })
        .collect::<Vec<_>>();
    let masses = vec![1.0f64 / n as f64; n];
    (positions, masses)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

fn skip_if_no_hip() -> bool {
    if !HipPmSolver::is_available() {
        eprintln!(
            "SKIP: HIP/ROCm no disponible (HIP_SKIP=1 o hipcc no encontrado). \
             Ejecutar con `-- --ignored` si ROCm está instalado."
        );
        return true;
    }
    false
}

#[test]
#[ignore = "Requiere hardware HIP/ROCm; ejecutar con `-- --ignored`"]
fn hip_pm_solver_creates_and_destroys() {
    if skip_if_no_hip() {
        return;
    }
    let solver = HipPmSolver::try_new(32, 100.0);
    assert!(
        solver.is_some(),
        "HipPmSolver::try_new devolvió None con HIP disponible"
    );
    drop(solver);
}

#[test]
#[ignore = "Requiere hardware HIP/ROCm; ejecutar con `-- --ignored`"]
fn hip_pm_accelerations_nonzero() {
    if skip_if_no_hip() {
        return;
    }
    let solver = HipPmSolver::try_new(32, 100.0).expect("HIP disponible");
    let box_size = 100.0f64;
    let n = 8;
    let (positions, masses) = test_particles(n, box_size);
    let indices: Vec<usize> = (0..n).collect();
    let mut out = vec![Vec3::zero(); n];

    solver.accelerations_for_indices(&positions, &masses, 0.01, 1.0, &indices, &mut out);

    let norm: f64 = out.iter().map(|v| v.x * v.x + v.y * v.y + v.z * v.z).sum();
    assert!(
        norm > 0.0,
        "Las aceleraciones PM HIP son todas cero — algo falló en el kernel"
    );
}

#[test]
#[ignore = "Requiere hardware HIP/ROCm; ejecutar con `-- --ignored`"]
fn hip_pm_empty_query_ok() {
    if skip_if_no_hip() {
        return;
    }
    let solver = HipPmSolver::try_new(32, 100.0).expect("HIP disponible");
    let (positions, masses) = test_particles(4, 100.0);
    let mut out: Vec<Vec3> = vec![];
    solver.accelerations_for_indices(&positions, &masses, 0.01, 1.0, &[], &mut out);
}

/// Test que verifica que el solver sin HIP disponible devuelve None.
/// Este test SIEMPRE corre (no requiere HIP).
#[test]
fn hip_pm_try_new_without_hip_returns_none_or_some() {
    let available = HipPmSolver::is_available();
    let solver = HipPmSolver::try_new(32, 100.0);
    assert_eq!(
        available,
        solver.is_some(),
        "is_available() y try_new() deben ser consistentes"
    );
}
