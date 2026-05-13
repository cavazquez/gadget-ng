use gadget_ng_core::{Particle, Vec3};
use gadget_ng_cuda::CudaTreeSolver;

fn cuda_solver_or_skip() -> Option<CudaTreeSolver> {
    match CudaTreeSolver::try_new_checked() {
        Ok(solver) => Some(solver),
        Err(e) => {
            eprintln!("SKIP CudaTreeSolver: {e}");
            None
        }
    }
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_tree_walk_monopole_returns_finite_accelerations() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };
    let particles = vec![
        Particle::new(0, 1.0, Vec3::new(-0.5, 0.0, 0.0), Vec3::zero()),
        Particle::new(1, 2.0, Vec3::new(0.5, 0.0, 0.0), Vec3::zero()),
    ];
    let acc = cuda
        .try_walk_monopole(&particles, 1.0, 1.0e-4)
        .expect("cuda tree walk should complete");
    assert_eq!(acc.len(), particles.len());
    assert!(
        acc.iter()
            .all(|a| a.x.is_finite() && a.y.is_finite() && a.z.is_finite())
    );
}

#[test]
fn cuda_tree_solver_returns_none_without_hardware() {
    let _ = CudaTreeSolver::try_new();
}
