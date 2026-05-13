use gadget_ng_core::{MolecularSection, Particle, Vec3};
use gadget_ng_cuda::CudaMolecularSolver;
use gadget_ng_sph::molecular_gas;

fn cuda_solver_or_skip() -> Option<CudaMolecularSolver> {
    match CudaMolecularSolver::try_new_checked() {
        Ok(solver) => Some(solver),
        Err(err) => {
            eprintln!("SKIP: {err}");
            None
        }
    }
}

fn assert_close_abs(label: &str, got: f64, expected: f64, tol: f64) {
    let diff = (got - expected).abs();
    assert!(
        diff <= tol,
        "{label}: got={got:.8e} expected={expected:.8e} diff={diff:.3e} tol={tol:.3e}"
    );
}

fn mol_cfg() -> MolecularSection {
    MolecularSection {
        enabled: true,
        rho_h2_threshold: 10.0,
        ..Default::default()
    }
}

fn gas_with_h2(n: usize) -> Vec<Particle> {
    let mut parts = Vec::with_capacity(n);
    for i in 0..n {
        let mut p = Particle::new_gas(i, 1.0, Vec3::zero(), Vec3::zero(), 0.0, 1.0);
        p.h2_fraction = 0.1;
        p.smoothing_length = 0.3;
        p.mass = 1.0;
        parts.push(p);
    }
    parts
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_h2_fraction_matches_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };
    let cfg = mol_cfg();
    let mut cpu = gas_with_h2(4);
    let mut gpu = cpu.clone();

    molecular_gas::update_h2_fraction(&mut cpu, &cfg, 0.05);
    cuda.try_update_h2(&mut gpu, &cfg, None, 0.05).unwrap();

    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_close_abs(&format!("h2[{i}]"), g.h2_fraction, c.h2_fraction, 2e-5);
    }
}

#[test]
fn cuda_molecular_solver_returns_none_without_hardware() {
    let result = CudaMolecularSolver::try_new();
    match result {
        None => eprintln!("CUDA no disponible (esperado en CI)."),
        Some(_) => eprintln!("CUDA disponible."),
    }
}
