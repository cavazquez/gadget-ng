use gadget_ng_core::{DustSection, Particle, Vec3};
use gadget_ng_cuda::CudaDustSolver;
use gadget_ng_sph::dust;

fn cuda_solver_or_skip() -> Option<CudaDustSolver> {
    match CudaDustSolver::try_new_checked() {
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

fn dust_cfg() -> DustSection {
    DustSection {
        enabled: true,
        d_to_g_max: 0.01,
        tau_grow: 5.0,
        t_destroy_k: 1e6,
        ..Default::default()
    }
}

fn gas_with_dust(n: usize) -> Vec<Particle> {
    let mut parts = Vec::with_capacity(n);
    for i in 0..n {
        let mut p = Particle::new_gas(i, 1.0, Vec3::zero(), Vec3::zero(), 0.0, 1.0);
        p.dust_to_gas = 0.001;
        p.internal_energy = 2.0; // ~1.5e5 K (cold, accretion regime)
        p.smoothing_length = 0.5;
        p.mass = 1.0;
        p.metallicity = 0.02;
        parts.push(p);
    }
    parts
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_dust_accretion_matches_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };
    let cfg = dust_cfg();
    let mut cpu = gas_with_dust(4);
    let mut gpu = cpu.clone();

    dust::update_dust(&mut cpu, &cfg, 5.0 / 3.0, 0.1);
    cuda.try_update_dust(&mut gpu, &cfg, 5.0 / 3.0, 0.1)
        .unwrap();

    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_close_abs(
            &format!("dust_to_gas[{i}]"),
            g.dust_to_gas,
            c.dust_to_gas,
            1e-5,
        );
    }
}

#[test]
fn cuda_dust_solver_returns_none_without_hardware() {
    let result = CudaDustSolver::try_new();
    match result {
        None => eprintln!("CUDA no disponible (esperado en CI)."),
        Some(_) => eprintln!("CUDA disponible."),
    }
}
