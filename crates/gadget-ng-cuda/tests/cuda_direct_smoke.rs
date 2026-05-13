//! Smoke tests para `CudaDirectGravity`.
//!
//! Ejecutar con hardware CUDA:
//!
//! ```bash
//! CUDA_ARCH=sm_61 cargo test -p gadget-ng-cuda --test cuda_direct_smoke -- --ignored --nocapture
//! ```

use gadget_ng_core::gravity::GravitySolver;
use gadget_ng_core::vec3::Vec3;
use gadget_ng_cuda::{CudaDirectGravity, CudaPmSolver};

fn skip_if_no_cuda() -> bool {
    if !CudaPmSolver::is_available() {
        eprintln!(
            "SKIP: CUDA no disponible ({}). Ejecutar con `-- --ignored` si CUDA está instalado.",
            CudaPmSolver::availability()
        );
        return true;
    }
    false
}

fn test_particles() -> (Vec<[f32; 3]>, Vec<f32>) {
    (
        vec![
            [0.10, 0.20, 0.30],
            [0.85, 0.25, 0.35],
            [0.35, 0.75, 0.55],
            [0.65, 0.60, 0.90],
        ],
        vec![1.0, 0.7, 1.3, 0.9],
    )
}

fn cpu_direct_reference(pos: &[[f32; 3]], mass: &[f32], eps: f32) -> Vec<[f32; 3]> {
    let eps2 = eps * eps;
    let mut out = vec![[0.0_f32; 3]; pos.len()];
    for i in 0..pos.len() {
        let mut ax = 0.0_f32;
        let mut ay = 0.0_f32;
        let mut az = 0.0_f32;
        for j in 0..pos.len() {
            let dx = pos[j][0] - pos[i][0];
            let dy = pos[j][1] - pos[i][1];
            let dz = pos[j][2] - pos[i][2];
            let r2 = dx * dx + dy * dy + dz * dz + eps2;
            let inv = 1.0 / r2.sqrt();
            let inv3 = inv * inv * inv;
            ax += mass[j] * dx * inv3;
            ay += mass[j] * dy * inv3;
            az += mass[j] * dz * inv3;
        }
        out[i] = [ax, ay, az];
    }
    out
}

fn assert_vec3_close(actual: [f32; 3], expected: [f32; 3]) {
    let err = ((actual[0] - expected[0]).powi(2)
        + (actual[1] - expected[1]).powi(2)
        + (actual[2] - expected[2]).powi(2))
    .sqrt();
    let scale = (expected[0] * expected[0] + expected[1] * expected[1] + expected[2] * expected[2])
        .sqrt()
        .max(1.0);
    assert!(
        err / scale < 2.0e-5,
        "CUDA direct mismatch: actual={actual:?} expected={expected:?} rel={:.3e}",
        err / scale
    );
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_direct_try_compute_matches_cpu_reference() {
    if skip_if_no_cuda() {
        return;
    }

    let eps = 0.03_f32;
    let (pos, mass) = test_particles();
    let solver = CudaDirectGravity::try_new_checked(eps).expect("CUDA disponible");
    let actual = solver
        .try_compute(&pos, &mass)
        .expect("cuda_direct_solve debe completar");
    let expected = cpu_direct_reference(&pos, &mass, eps);

    for (actual_i, expected_i) in actual.into_iter().zip(expected) {
        assert_vec3_close(actual_i, expected_i);
    }
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_direct_gravity_solver_bridge_supports_partial_indices() {
    if skip_if_no_cuda() {
        return;
    }

    let eps = 0.03_f32;
    let (pos_f32, mass_f32) = test_particles();
    let positions = pos_f32
        .iter()
        .map(|p| Vec3::new(p[0] as f64, p[1] as f64, p[2] as f64))
        .collect::<Vec<_>>();
    let masses = mass_f32.iter().map(|&m| m as f64).collect::<Vec<_>>();
    let query = vec![3_usize, 1_usize];
    let mut actual = vec![Vec3::zero(); query.len()];

    let solver = CudaDirectGravity::try_new_checked(eps).expect("CUDA disponible");
    solver.accelerations_for_indices(
        &positions,
        &masses,
        eps as f64 * eps as f64,
        1.0,
        &query,
        &mut actual,
    );

    let expected_all = cpu_direct_reference(&pos_f32, &mass_f32, eps);
    for (k, &gi) in query.iter().enumerate() {
        assert_vec3_close(
            [actual[k].x as f32, actual[k].y as f32, actual[k].z as f32],
            expected_all[gi],
        );
    }
}
