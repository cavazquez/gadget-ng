use gadget_ng_core::{Particle, Vec3};
use gadget_ng_cuda::CudaRtSolver;
use gadget_ng_rt::{
    coupling::{apply_photoheating, photoionization_rate},
    m1::{M1Params, RadiationField},
};

fn rt_field() -> RadiationField {
    let mut rad = RadiationField::uniform(4, 4, 4, 1.0, 2.0);
    for i in 0..rad.n_cells() {
        rad.energy_density[i] = 1.0 + 0.01 * i as f64;
        rad.flux_x[i] = 0.001 * i as f64;
        rad.flux_y[i] = 0.002;
        rad.flux_z[i] = -0.001;
    }
    rad
}

fn particles() -> Vec<Particle> {
    (0..16)
        .map(|i| {
            Particle::new_gas(
                i,
                1.0,
                Vec3::new((i % 4) as f64 + 0.1, ((i / 4) % 4) as f64 + 0.1, 0.1),
                Vec3::zero(),
                1.0,
                0.2,
            )
        })
        .collect()
}

fn cuda_solver_or_skip() -> Option<CudaRtSolver> {
    match CudaRtSolver::try_new_checked() {
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
fn cuda_rt_field_diagnostics_match_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };
    let rad = rt_field();
    let params = M1Params::default();
    let (e_gpu, xi_gpu, gamma_gpu) = cuda.try_field_diagnostics(&rad, &params, 1.0).unwrap();
    let e_cpu = rad.total_energy(1.0);
    let xi_cpu = rad.xi_field(gadget_ng_rt::m1::C_KMS / params.c_red_factor);
    let gamma_cpu = photoionization_rate(&rad, &params);

    assert_close_rel("total_energy", e_gpu, e_cpu, 1.0e-6);
    for i in 0..rad.n_cells() {
        assert_close_rel(&format!("xi[{i}]"), xi_gpu[i], xi_cpu[i], 1.0e-5);
        assert_close_rel(&format!("gamma[{i}]"), gamma_gpu[i], gamma_cpu[i], 1.0e-5);
    }
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_rt_photoheating_matches_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };
    let rad = rt_field();
    let params = M1Params::default();
    let gamma = photoionization_rate(&rad, &params);
    let mut cpu = particles();
    let mut gpu = cpu.clone();

    apply_photoheating(&mut cpu, &rad, &gamma, 0.5, 4.0);
    cuda.try_apply_photoheating(&mut gpu, &rad, &gamma, 0.5, 4.0)
        .unwrap();

    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_close_rel(
            &format!("internal_energy[{i}]"),
            g.internal_energy,
            c.internal_energy,
            1.0e-6,
        );
    }
}
