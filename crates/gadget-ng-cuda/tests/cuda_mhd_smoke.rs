use gadget_ng_core::{Particle, Vec3};
use gadget_ng_cuda::CudaMhdSolver;
use gadget_ng_mhd::{apply_flux_freeze, b_field_stats, mean_gas_density};

fn particles() -> Vec<Particle> {
    (0..32)
        .map(|i| {
            let mut p = Particle::new_gas(
                i,
                1.0 + 0.01 * i as f64,
                Vec3::new(i as f64, 0.0, 0.0),
                Vec3::zero(),
                0.5 + 0.02 * i as f64,
                0.8 + 0.01 * i as f64,
            );
            p.b_field = Vec3::new(
                1.0e-4 * (i as f64 + 1.0),
                2.0e-4,
                -1.0e-4 * (i as f64 + 0.5),
            );
            p
        })
        .collect()
}

fn cuda_solver_or_skip() -> Option<CudaMhdSolver> {
    match CudaMhdSolver::try_new_checked() {
        Ok(solver) => Some(solver),
        Err(err) => {
            eprintln!("SKIP: {err}");
            None
        }
    }
}

fn assert_close_rel(label: &str, got: f64, expected: f64, tol: f64) {
    let denom = expected.abs().max(1.0e-12);
    let rel = (got - expected).abs() / denom;
    assert!(
        rel <= tol,
        "{label}: got={got:.8e} expected={expected:.8e} rel={rel:.3e} tol={tol:.3e}"
    );
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_mhd_mean_density_matches_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };
    let parts = particles();
    let cpu = mean_gas_density(&parts);
    let gpu = cuda.try_mean_gas_density(&parts).unwrap();
    assert_close_rel("mean density", gpu, cpu, 2.0e-6);
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_mhd_b_stats_match_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };
    let parts = particles();
    let cpu = b_field_stats(&parts).unwrap();
    let gpu = cuda.try_b_field_stats(&parts).unwrap().unwrap();
    assert_close_rel("b_mean", gpu.b_mean, cpu.b_mean, 5.0e-5);
    assert_close_rel("b_rms", gpu.b_rms, cpu.b_rms, 5.0e-5);
    assert_close_rel("b_max", gpu.b_max, cpu.b_max, 5.0e-5);
    assert_close_rel("e_mag", gpu.e_mag, cpu.e_mag, 5.0e-5);
    assert_eq!(gpu.n_gas, cpu.n_gas);
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_mhd_flux_freeze_matches_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };
    let mut cpu = particles();
    let mut gpu = cpu.clone();
    let rho_ref = mean_gas_density(&cpu);
    apply_flux_freeze(&mut cpu, 5.0 / 3.0, 1.0e12, rho_ref);
    cuda.try_apply_flux_freeze(&mut gpu, 5.0 / 3.0, 1.0e12, rho_ref)
        .unwrap();

    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_close_rel(&format!("bx[{i}]"), g.b_field.x, c.b_field.x, 5.0e-5);
        assert_close_rel(&format!("by[{i}]"), g.b_field.y, c.b_field.y, 5.0e-5);
        assert_close_rel(&format!("bz[{i}]"), g.b_field.z, c.b_field.z, 5.0e-5);
    }
}
