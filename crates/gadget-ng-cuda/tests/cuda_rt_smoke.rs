use gadget_ng_core::{Particle, Vec3};
use gadget_ng_cuda::CudaRtSolver;
use gadget_ng_rt::{
    coupling::{apply_photoheating, photoionization_rate},
    m1::{M1Params, RadiationField, m1_update},
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

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_rt_m1_advection_matches_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };

    // Campo con perturbación sinusoidal de energía + flujo no nulo.
    let nx = 8;
    let ny = 8;
    let nz = 8;
    let dx = 1.0_f64;
    let mut cpu = RadiationField::uniform(nx, ny, nz, dx, 1.0);
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let i = cpu.idx(ix, iy, iz);
                let theta = 2.0 * std::f64::consts::PI * ix as f64 / nx as f64;
                cpu.energy_density[i] = 1.0 + 0.05 * theta.sin();
                cpu.flux_x[i] = 0.01 * theta.cos();
                cpu.flux_y[i] = 0.005 * theta.sin();
                cpu.flux_z[i] = 0.002;
            }
        }
    }
    let mut gpu = cpu.clone();

    let params = M1Params {
        c_red_factor: 10.0,
        kappa_abs: 0.0,
        kappa_scat: 0.0,
        substeps: 5,
        ..Default::default()
    };
    let dt = 0.05;

    m1_update(&mut cpu, dt, &params);
    cuda.try_m1_advection(&mut gpu, dt, &params).unwrap();

    // f32 vs f64: tolerancia 1e-4 en energía total y 1e-3 en celdas individuales.
    let e_cpu = cpu.total_energy(1.0);
    let e_gpu = gpu.total_energy(1.0);
    assert_close_rel("m1_total_energy", e_gpu, e_cpu, 1.0e-4);

    for i in 0..cpu.n_cells() {
        assert_close_rel(
            &format!("m1_e[{i}]"),
            gpu.energy_density[i],
            cpu.energy_density[i],
            1.0e-3,
        );
        assert!(gpu.energy_density[i] >= 0.0, "m1_e[{i}] debe ser >= 0");
    }
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_rt_chemistry_rates_match_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };

    // 128 partículas gas con posiciones uniformes en una caja 4x4x4 con 8x8x8 celdas
    let n = 128_usize;
    let nx = 8_usize;
    let box_size = 8.0_f64;
    let dx = box_size / nx as f64;
    let rad = RadiationField::uniform(nx, nx, nx, dx, box_size);
    let params = M1Params::default();
    let particles: Vec<Particle> = (0..n)
        .map(|i| {
            Particle::new_gas(
                i,
                1.0,
                Vec3::new(
                    (i % nx) as f64 + 0.3,
                    ((i / nx) % nx) as f64 + 0.3,
                    (i / (nx * nx)) as f64 + 0.3,
                ),
                Vec3::zero(),
                1.0,
                0.3,
            )
        })
        .collect();

    let gamma_gpu = cuda
        .try_chemistry_rates(&particles, &rad, &params, box_size)
        .unwrap();
    assert_eq!(gamma_gpu.len(), n);

    // CPU: NGP lookup manual
    use gadget_ng_rt::m1::C_KMS;
    let c_red_cgs = C_KMS * 1.0e5 / params.c_red_factor;
    let sigma_hi: f64 = 6.3e-18;
    let h_nu_0: f64 = 2.179e-11;
    for (i, p) in particles.iter().enumerate() {
        let ix = ((p.position.x / box_size * nx as f64) as usize).min(nx - 1);
        let iy = ((p.position.y / box_size * nx as f64) as usize).min(nx - 1);
        let iz = ((p.position.z / box_size * nx as f64) as usize).min(nx - 1);
        let e = rad.energy_density[rad.idx(ix, iy, iz)].max(0.0);
        let gamma_cpu = sigma_hi * c_red_cgs * e / h_nu_0;
        assert_close_rel(&format!("gamma_hi[{i}]"), gamma_gpu[i], gamma_cpu, 1.0e-4);
    }
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_rt_chemistry_stiff_match_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };

    use gadget_ng_core::ParticleType;
    use gadget_ng_rt::chemistry::{ChemParams, ChemState, apply_chemistry};
    use gadget_ng_rt::m1::RadiationField;

    let n = 128_usize;
    let rad = RadiationField::uniform(4, 4, 4, 0.25, 1.0);
    let chem_params = ChemParams::default();
    let dt = 1.0e4_f64;

    let mut states_cpu: Vec<ChemState> = (0..n).map(|_| ChemState::neutral()).collect();
    let mut states_gpu = states_cpu.clone();
    let mut particles_cpu: Vec<Particle> = (0..n)
        .map(|i| {
            Particle::new_gas(
                i,
                1.0,
                Vec3::new((i % 4) as f64 * 0.25, ((i / 4) % 4) as f64 * 0.25, 0.1),
                Vec3::zero(),
                1.0,
                0.2,
            )
        })
        .collect();

    // CPU
    apply_chemistry(&mut particles_cpu, &mut states_cpu, &rad, &chem_params, dt);

    // GPU: primero computar gamma_hi usando try_chemistry_rates (consistente con rad)
    use gadget_ng_rt::m1::M1Params;
    let m1_params = M1Params::default();
    let gamma_hi = cuda
        .try_chemistry_rates(&particles_cpu, &rad, &m1_params, 1.0)
        .unwrap();
    let temperature: Vec<f64> = particles_cpu
        .iter()
        .map(|p| {
            ChemState::neutral()
                .temperature_from_internal_energy(p.internal_energy, chem_params.gamma)
        })
        .collect();
    let ptypes: Vec<ParticleType> = particles_cpu.iter().map(|p| p.ptype).collect();
    cuda.try_apply_chemistry(
        &mut states_gpu,
        &gamma_hi,
        &temperature,
        &ptypes,
        dt,
        chem_params.n_h_ref,
    )
    .unwrap();

    // Tolerancia 5% (f32 vs f64 + divergencia de sub-pasos adaptativos)
    for i in 0..n {
        assert_close_rel(
            &format!("x_hii[{i}]"),
            states_gpu[i].x_hii,
            states_cpu[i].x_hii,
            0.05,
        );
        assert_close_rel(
            &format!("x_e[{i}]"),
            states_gpu[i].x_e,
            states_cpu[i].x_e,
            0.05,
        );
    }
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_rt_reionization_stats_match_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };

    use gadget_ng_rt::chemistry::ChemState;
    use gadget_ng_rt::reionization::compute_reionization_state;

    let n = 256_usize;
    let states: Vec<ChemState> = (0..n)
        .map(|i| ChemState {
            x_hi: 0.5,
            x_hii: 0.1 + 0.003 * i as f64,
            x_hei: 0.07,
            x_heii: 0.005,
            x_heiii: 0.001,
            x_e: 0.12,
            x_hm: 0.0,
            x_h2: 0.0,
            x_h2p: 0.0,
            x_d: 2.5e-5,
            x_dp: 0.0,
            x_hd: 0.0,
        })
        .collect();

    let z = 8.5;
    let n_sources = 10_usize;
    let rs_gpu = cuda.try_reionization_stats(&states, z, n_sources).unwrap();
    let rs_cpu = compute_reionization_state(&states, z, n_sources);

    assert_close_rel("x_hii_mean", rs_gpu.x_hii_mean, rs_cpu.x_hii_mean, 1.0e-4);
    assert_close_rel(
        "x_hii_sigma",
        rs_gpu.x_hii_sigma,
        rs_cpu.x_hii_sigma,
        1.0e-3,
    );
    assert_close_rel(
        "ionized_volume_fraction",
        rs_gpu.ionized_volume_fraction,
        rs_cpu.ionized_volume_fraction,
        1.0e-4,
    );
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_rt_cm21_field_match_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };

    use gadget_ng_rt::chemistry::ChemState;
    use gadget_ng_rt::cm21::{Cm21Params, brightness_temperature};

    let n = 256_usize;
    let states: Vec<ChemState> = (0..n)
        .map(|i| ChemState {
            x_hi: 0.5,
            x_hii: 0.1 + 0.003 * i as f64,
            ..ChemState::neutral()
        })
        .collect();
    let overdensity: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * i as f64).collect();
    let z = 8.5_f64;
    let params = Cm21Params::default();

    let dtb_gpu = cuda.try_cm21_field(&states, &overdensity, z).unwrap();

    // CPU: brightness_temperature por partícula con overdensity dado
    let dtb_cpu: Vec<f64> = states
        .iter()
        .zip(overdensity.iter())
        .map(|(s, &od)| brightness_temperature(s.x_hii, od, z, &params))
        .collect();

    assert_eq!(dtb_gpu.len(), n);
    for i in 0..n {
        assert_close_rel(&format!("delta_tb[{i}]"), dtb_gpu[i], dtb_cpu[i], 1.0e-4);
    }
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_rt_igm_temp_match_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };

    use gadget_ng_core::{Particle, Vec3};
    use gadget_ng_rt::chemistry::ChemState;
    use gadget_ng_rt::{IgmTempParams, compute_igm_temp_profile};

    let n = 512_usize;
    let particles: Vec<Particle> = (0..n)
        .map(|i| {
            let mut p = Particle::new_gas(
                i,
                (i + 1) as f64 * 0.001,
                Vec3::new(i as f64 * 0.01, 0.0, 0.0),
                Vec3::zero(),
                0.05,
                0.1,
            );
            p.internal_energy = 0.5 + 0.01 * (i % 50) as f64;
            p
        })
        .collect();
    let chem: Vec<ChemState> = (0..n)
        .map(|i| ChemState {
            x_hii: 0.5 + 0.001 * (i % 100) as f64,
            x_hi: 0.5 - 0.001 * (i % 100) as f64,
            ..ChemState::neutral()
        })
        .collect();
    // delta_max = 0.0 deshabilita el filtro IGM para este test de paridad.
    let igm_params = IgmTempParams {
        delta_max: 0.0,
        gamma: 5.0 / 3.0,
    };
    let mean_density = 0.0_f64;
    let z = 7.5_f64;

    let gpu_bin = cuda
        .try_igm_temp_profile(&particles, &chem, mean_density, z, &igm_params)
        .unwrap();
    let cpu_bin = compute_igm_temp_profile(&particles, &chem, mean_density, z, &igm_params);

    assert_eq!(gpu_bin.z, z);
    assert!(cpu_bin.n_particles > 0, "CPU no encontró partículas IGM");
    // n_particles deben coincidir exactamente (sin filtro, todos incluidos)
    assert_eq!(
        gpu_bin.n_particles, cpu_bin.n_particles,
        "n_igm discrepante: gpu={} cpu={}",
        gpu_bin.n_particles, cpu_bin.n_particles
    );
    // t_mean: tolerancia 2% (f32 vs f64)
    assert_close_rel("t_mean", gpu_bin.t_mean, cpu_bin.t_mean, 0.02);
    // t_sigma: tolerancia mayor por acumulación f32
    assert_close_rel("t_sigma", gpu_bin.t_sigma, cpu_bin.t_sigma, 0.10);
}

// ── AP-17: IGM temp percentiles desde array compacto GPU ─────────────────────

#[test]
#[ignore]
fn cuda_rt_igm_temp_percentiles_match_cpu() {
    use gadget_ng_rt::{IgmTempParams, compute_igm_temp_profile};

    let cuda = match CudaRtSolver::try_new_checked() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP: {e}");
            return;
        }
    };

    let n = 64_usize;
    let particles: Vec<Particle> = (0..n)
        .map(|i| {
            Particle::new_gas(
                i,
                0.01,
                Vec3::new(i as f64 * 0.1, (i % 8) as f64 * 0.1, 0.0),
                Vec3::zero(),
                0.5,
                0.5 + 0.02 * i as f64, // varied u → varied T
            )
        })
        .collect();
    let chem: Vec<gadget_ng_rt::ChemState> = vec![gadget_ng_rt::ChemState::neutral(); n];
    let igm_params = IgmTempParams {
        delta_max: 0.0,
        gamma: 5.0 / 3.0,
    };
    let mean_density = 0.0_f64;
    let z = 6.0_f64;

    let gpu_bin = cuda
        .try_igm_temp_profile(&particles, &chem, mean_density, z, &igm_params)
        .unwrap();
    let cpu_bin = compute_igm_temp_profile(&particles, &chem, mean_density, z, &igm_params);

    assert!(cpu_bin.n_particles > 0, "CPU: no hay partículas IGM");
    assert_eq!(
        gpu_bin.n_particles, cpu_bin.n_particles,
        "n_igm: gpu={} cpu={}",
        gpu_bin.n_particles, cpu_bin.n_particles
    );

    // Verificar que los percentiles son coherentes (GPU ordena el array compacto)
    // Tolerancia del 5% para percentiles (f32 vs f64)
    let tol = 0.05_f64;
    for (label, gpu_val, cpu_val) in [
        ("t_median", gpu_bin.t_median, cpu_bin.t_median),
        ("t_p16", gpu_bin.t_p16, cpu_bin.t_p16),
        ("t_p84", gpu_bin.t_p84, cpu_bin.t_p84),
    ] {
        let rel = (gpu_val - cpu_val).abs() / cpu_val.abs().max(1.0e-10);
        assert!(
            rel < tol,
            "{label} GPU={gpu_val:.4e} CPU={cpu_val:.4e} rel={rel:.4}"
        );
    }
    eprintln!(
        "[cuda_rt_igm_temp_percentiles_match_cpu] OK — median GPU={:.2e} CPU={:.2e}",
        gpu_bin.t_median, cpu_bin.t_median
    );
}
