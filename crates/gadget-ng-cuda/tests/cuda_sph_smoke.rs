use gadget_ng_core::Vec3;
use gadget_ng_cuda::CudaSphSolver;
use gadget_ng_sph::{
    compute_balsara_factors_with_periodic, compute_density_with_periodic,
    compute_sph_forces_gadget2_with_periodic, compute_sph_forces_with_periodic,
    particle::SphParticle,
};

fn glass_particles(n_side: usize, box_size: f64) -> Vec<SphParticle> {
    let dx = box_size / n_side as f64;
    let mut parts = Vec::with_capacity(n_side.pow(3));
    let mut id = 0usize;
    for ix in 0..n_side {
        for iy in 0..n_side {
            for iz in 0..n_side {
                let x = (ix as f64 + 0.5) * dx;
                let y = (iy as f64 + 0.5) * dx;
                let z = (iz as f64 + 0.5) * dx;
                let vx = 0.02 * (y - 0.5 * box_size);
                let vy = -0.01 * (x - 0.5 * box_size);
                let vz = 0.005 * (z - 0.5 * box_size);
                parts.push(SphParticle::new_gas(
                    id,
                    1.0,
                    Vec3::new(x, y, z),
                    Vec3::new(vx, vy, vz),
                    1.0,
                    2.0 * dx,
                ));
                id += 1;
            }
        }
    }
    parts
}

fn cuda_solver_or_skip() -> Option<CudaSphSolver> {
    match CudaSphSolver::try_new_checked() {
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
fn cuda_sph_density_matches_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };
    let box_size = 4.0;
    let mut cpu = glass_particles(4, box_size);
    let mut gpu = cpu.clone();

    compute_density_with_periodic(&mut cpu, Some(box_size));
    cuda.try_compute_density(&mut gpu, Some(box_size)).unwrap();

    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let cgas = c.gas.as_ref().unwrap();
        let ggas = g.gas.as_ref().unwrap();
        assert_close_rel(&format!("rho[{i}]"), ggas.rho, cgas.rho, 3.0e-3);
        assert_close_rel(
            &format!("pressure[{i}]"),
            ggas.pressure,
            cgas.pressure,
            3.0e-3,
        );
        assert_close_rel(&format!("h_sml[{i}]"), ggas.h_sml, cgas.h_sml, 3.0e-3);
    }
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_sph_balsara_matches_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };
    let box_size = 4.0;
    let mut cpu = glass_particles(4, box_size);
    compute_density_with_periodic(&mut cpu, Some(box_size));
    let mut gpu = cpu.clone();

    compute_balsara_factors_with_periodic(&mut cpu, Some(box_size));
    cuda.try_compute_balsara(&mut gpu, Some(box_size)).unwrap();

    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let cgas = c.gas.as_ref().unwrap();
        let ggas = g.gas.as_ref().unwrap();
        assert_close_rel(&format!("balsara[{i}]"), ggas.balsara, cgas.balsara, 2.0e-2);
    }
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_sph_forces_match_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };
    let box_size = 4.0;
    let mut cpu = glass_particles(4, box_size);
    compute_density_with_periodic(&mut cpu, Some(box_size));
    let mut gpu = cpu.clone();

    compute_sph_forces_with_periodic(&mut cpu, Some(box_size));
    cuda.try_compute_forces(&mut gpu, Some(box_size)).unwrap();

    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let cgas = c.gas.as_ref().unwrap();
        let ggas = g.gas.as_ref().unwrap();
        assert_close_rel(
            &format!("acc_x[{i}]"),
            ggas.acc_sph.x,
            cgas.acc_sph.x,
            3.0e-2,
        );
        assert_close_rel(
            &format!("acc_y[{i}]"),
            ggas.acc_sph.y,
            cgas.acc_sph.y,
            3.0e-2,
        );
        assert_close_rel(
            &format!("acc_z[{i}]"),
            ggas.acc_sph.z,
            cgas.acc_sph.z,
            3.0e-2,
        );
        assert_close_rel(&format!("du_dt[{i}]"), ggas.du_dt, cgas.du_dt, 3.0e-2);
    }
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_sph_gadget2_forces_match_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };
    let box_size = 4.0;
    let mut cpu = glass_particles(4, box_size);
    compute_density_with_periodic(&mut cpu, Some(box_size));
    compute_balsara_factors_with_periodic(&mut cpu, Some(box_size));
    let mut gpu = cpu.clone();

    compute_sph_forces_gadget2_with_periodic(&mut cpu, Some(box_size));
    cuda.try_compute_gadget2_forces(&mut gpu, Some(box_size))
        .unwrap();

    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let cgas = c.gas.as_ref().unwrap();
        let ggas = g.gas.as_ref().unwrap();
        assert_close_rel(
            &format!("g2_acc_x[{i}]"),
            ggas.acc_sph.x,
            cgas.acc_sph.x,
            3.0e-2,
        );
        assert_close_rel(
            &format!("g2_acc_y[{i}]"),
            ggas.acc_sph.y,
            cgas.acc_sph.y,
            3.0e-2,
        );
        assert_close_rel(
            &format!("g2_acc_z[{i}]"),
            ggas.acc_sph.z,
            cgas.acc_sph.z,
            3.0e-2,
        );
        assert_close_rel(&format!("g2_da_dt[{i}]"), ggas.da_dt, cgas.da_dt, 3.0e-2);
        assert_close_rel(
            &format!("g2_max_vsig[{i}]"),
            ggas.max_vsig,
            cgas.max_vsig,
            3.0e-2,
        );
    }
}
