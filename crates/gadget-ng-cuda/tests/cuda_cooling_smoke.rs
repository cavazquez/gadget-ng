use gadget_ng_core::{CoolingKind, Particle, SphSection, Vec3};
use gadget_ng_cuda::CudaCoolingSolver;
use gadget_ng_sph::apply_cooling_with_redshift;

fn cuda_solver_or_skip() -> Option<CudaCoolingSolver> {
    match CudaCoolingSolver::try_new_checked() {
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

fn cfg_atomic_hhe() -> SphSection {
    let mut cfg = SphSection::default();
    cfg.gamma = 5.0 / 3.0;
    cfg.t_floor_k = 1e4;
    cfg.cooling = CoolingKind::AtomicHHe;
    cfg
}

fn hot_gas_particles(n: usize) -> Vec<Particle> {
    let mut parts = Vec::with_capacity(n);
    for i in 0..n {
        let mut p = Particle::new_gas(i, 1.0, Vec3::zero(), Vec3::zero(), 0.0, 1.0);
        p.internal_energy = 5.0; // ~3.6e5 K
        p.smoothing_length = 0.5;
        p.mass = 1.0;
        parts.push(p);
    }
    parts
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_cooling_atomic_matches_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };
    let cfg = cfg_atomic_hhe();
    let mut cpu = hot_gas_particles(4);
    let mut gpu = cpu.clone();

    apply_cooling_with_redshift(&mut cpu, &cfg, 0.01, 0.0);
    cuda.try_apply_cooling(&mut gpu, &cfg, 0.01, 0.0, 0.0)
        .unwrap();

    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_close_abs(
            &format!("u[{i}]"),
            g.internal_energy,
            c.internal_energy,
            2e-4,
        );
    }
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_cooling_fails_on_empty() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };
    let cfg = cfg_atomic_hhe();
    let mut empty: Vec<Particle> = Vec::new();
    assert!(
        cuda.try_apply_cooling(&mut empty, &cfg, 0.01, 0.0, 0.0)
            .is_ok()
    );
}

#[test]
fn cuda_cooling_solver_returns_none_without_hardware() {
    let result = CudaCoolingSolver::try_new();
    match result {
        None => eprintln!("CUDA no disponible (esperado en CI)."),
        Some(_) => eprintln!("CUDA disponible."),
    }
}
