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

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_mhd_cr_streaming_match_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };
    use gadget_ng_mhd::streaming_crk;

    // N=64 partículas gas con cr_energy > 0 y campo B
    let n = 64_usize;
    let mut cpu: Vec<Particle> = (0..n)
        .map(|i| {
            let mut p = Particle::new_gas(
                i,
                1.0,
                Vec3::new(i as f64 * 0.1, 0.1, 0.1),
                Vec3::new(0.001 * i as f64, 0.0, 0.0),
                0.1 + 0.002 * i as f64,
                0.3,
            );
            p.cr_energy = 0.01 + 0.001 * i as f64;
            p.b_field = Vec3::new(1.0e-4, 2.0e-4, 0.0);
            p
        })
        .collect();
    let mut gpu = cpu.clone();

    let dt = 0.5;
    let coeff = 0.1;
    let pbox = Some(10.0);

    streaming_crk(&mut cpu, dt, coeff, pbox);
    cuda.try_cr_streaming(&mut gpu, dt, coeff, pbox).unwrap();

    // Tolerancia 5%: f32 vs f64 + divergencia de div_v SPH en f32
    let tol = 0.05;
    for i in 0..n {
        let rel = (gpu[i].cr_energy - cpu[i].cr_energy).abs()
            / cpu[i].cr_energy.abs().max(1.0e-20);
        assert!(
            rel <= tol,
            "cr_energy[{i}]: gpu={:.6e} cpu={:.6e} rel={:.3e}",
            gpu[i].cr_energy,
            cpu[i].cr_energy,
            rel
        );
    }
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_mhd_cr_backreaction_match_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };
    use gadget_ng_mhd::cr_pressure_backreaction;

    let n = 64_usize;
    let parts: Vec<Particle> = (0..n)
        .map(|i| {
            let mut p = Particle::new_gas(
                i,
                1.0,
                Vec3::new(i as f64 * 0.15, 0.1, 0.1),
                Vec3::zero(),
                0.1 + 0.003 * i as f64,
                0.3,
            );
            p.cr_energy = 0.01 + 0.002 * i as f64;
            p
        })
        .collect();
    let mut cpu = parts.clone();

    // CPU: aplica backreaction in-place
    cr_pressure_backreaction(&mut cpu, Some(10.0));

    // GPU: devuelve aceleraciones
    let accel_gpu = cuda.try_cr_backreaction(&parts, Some(10.0)).unwrap();
    assert_eq!(accel_gpu.len(), n);

    // Para partículas con cr_energy > 0 en el rango central, compara magnitudes
    let tol = 0.05;
    for i in 0..n {
        if parts[i].cr_energy < 1.0e-15 {
            continue;
        }
        let da_cpu = (cpu[i].acceleration.x - parts[i].acceleration.x).hypot(
            (cpu[i].acceleration.y - parts[i].acceleration.y)
                .hypot(cpu[i].acceleration.z - parts[i].acceleration.z),
        );
        let da_gpu = accel_gpu[i]
            .x
            .hypot(accel_gpu[i].y.hypot(accel_gpu[i].z));
        let denom = da_cpu.abs().max(1.0e-30);
        let rel = (da_gpu - da_cpu).abs() / denom;
        assert!(
            rel <= tol || da_cpu < 1.0e-20,
            "cr_backreaction[{i}]: |a_gpu|={:.6e} |a_cpu|={:.6e} rel={:.3e}",
            da_gpu,
            da_cpu,
            rel
        );
    }
}
