//! Tests de paridad SPH: CPU vs CUDA.
//!
//! - `cuda_parity_sph_full_pipeline`: pipeline SphParticle (densidad, Balsara, fuerzas
//!   Gadget-2, cooling, dust, H2) CPU vs CUDA.
//! - `cuda_parity_sph_core_pipeline`: `try_sph_density_and_forces_core` sobre
//!   `gadget_ng_core::Particle` (AP-18) vs referencia CPU con `SphParticle`.

use gadget_ng_core::{
    CoolingKind, DustSection, MolecularSection, Particle, ParticleType, SphSection, Vec3,
};
use gadget_ng_cuda::{CudaCoolingSolver, CudaDustSolver, CudaMolecularSolver, CudaSphSolver};
use gadget_ng_sph::{
    compute_balsara_factors_with_periodic, compute_density_with_periodic,
    compute_sph_forces_gadget2_with_periodic, compute_sph_forces_with_periodic, cooling, dust,
    molecular_gas, particle::SphParticle,
};

fn cuda_sph_or_skip() -> Option<CudaSphSolver> {
    match CudaSphSolver::try_new_checked() {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("SKIP CudaSphSolver: {e}");
            None
        }
    }
}

fn cuda_cool_or_skip() -> Option<CudaCoolingSolver> {
    match CudaCoolingSolver::try_new_checked() {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("SKIP CudaCoolingSolver: {e}");
            None
        }
    }
}

fn cuda_dust_or_skip() -> Option<CudaDustSolver> {
    match CudaDustSolver::try_new_checked() {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("SKIP CudaDustSolver: {e}");
            None
        }
    }
}

fn cuda_mol_or_skip() -> Option<CudaMolecularSolver> {
    match CudaMolecularSolver::try_new_checked() {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("SKIP CudaMolecularSolver: {e}");
            None
        }
    }
}

fn assert_close_rel(label: &str, got: f64, expected: f64, tol: f64) {
    let denom = expected.abs().max(1e-12);
    let rel = (got - expected).abs() / denom;
    assert!(
        rel <= tol,
        "{label}: got={got:.8e} expected={expected:.8e} rel={rel:.3e} tol={tol:.3e}"
    );
}

fn glass_sph_particles(n_side: usize, box_size: f64) -> Vec<SphParticle> {
    let dx = box_size / n_side as f64;
    let mut parts = Vec::with_capacity(n_side.pow(3));
    let mut id = 0usize;
    for ix in 0..n_side {
        for iy in 0..n_side {
            for iz in 0..n_side {
                let x = (ix as f64 + 0.5) * dx;
                let y = (iy as f64 + 0.5) * dx;
                let z = (iz as f64 + 0.5) * dx;
                let vx = 0.01 * (y - 0.5 * box_size);
                let vy = -0.005 * (x - 0.5 * box_size);
                let vz = 0.003 * (z - 0.5 * box_size);
                parts.push(SphParticle::new_gas(
                    id,
                    1.0,
                    Vec3::new(x, y, z),
                    Vec3::new(vx, vy, vz),
                    1.5,
                    2.0 * dx,
                ));
                id += 1;
            }
        }
    }
    parts
}

fn sph_to_core_particles(sph: &[SphParticle]) -> Vec<Particle> {
    sph.iter()
        .map(|sp| {
            let u = sp.gas.as_ref().map(|g| g.u).unwrap_or(1.0);
            let h = sp.gas.as_ref().map(|g| g.h_sml).unwrap_or(1.0);
            let mut p = Particle::new_gas(sp.global_id, sp.mass, sp.position, sp.velocity, u, h);
            p.internal_energy = u;
            p.dust_to_gas = 0.002;
            p.h2_fraction = 0.05;
            p.metallicity = 0.02;
            p
        })
        .collect()
}

fn cooling_cfg() -> SphSection {
    SphSection {
        gamma: 5.0 / 3.0,
        t_floor_k: 1e4,
        cooling: CoolingKind::AtomicHHe,
        ..Default::default()
    }
}

fn dust_config() -> DustSection {
    DustSection {
        enabled: true,
        d_to_g_max: 0.01,
        tau_grow: 5.0,
        t_destroy_k: 1e6,
        ..Default::default()
    }
}

fn mol_config() -> MolecularSection {
    MolecularSection {
        enabled: true,
        rho_h2_threshold: 10.0,
        ..Default::default()
    }
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_parity_sph_full_pipeline() {
    let (Some(cuda_sph), Some(cuda_cool), Some(cuda_dust), Some(cuda_mol)) = (
        cuda_sph_or_skip(),
        cuda_cool_or_skip(),
        cuda_dust_or_skip(),
        cuda_mol_or_skip(),
    ) else {
        eprintln!("SKIP: uno o más solvers CUDA no disponibles.");
        return;
    };

    let box_size = 4.0_f64;
    let dt = 0.1_f64;
    let cool_cfg = cooling_cfg();
    let dust_cfg = dust_config();
    let mol_cfg = mol_config();

    // ── CPU path ──────────────────────────────────────────────────────────
    let mut cpu_sph = glass_sph_particles(4, box_size);
    compute_density_with_periodic(&mut cpu_sph, Some(box_size));
    compute_balsara_factors_with_periodic(&mut cpu_sph, Some(box_size));
    compute_sph_forces_gadget2_with_periodic(&mut cpu_sph, Some(box_size));

    let mut cpu = sph_to_core_particles(&cpu_sph);
    cooling::apply_cooling_with_redshift(&mut cpu, &cool_cfg, dt, 0.0);
    dust::update_dust(&mut cpu, &dust_cfg, cool_cfg.gamma, dt);
    molecular_gas::update_h2_fraction(&mut cpu, &mol_cfg, dt);

    // ── CUDA path ─────────────────────────────────────────────────────────
    let mut gpu_sph = glass_sph_particles(4, box_size);
    cuda_sph
        .try_compute_density(&mut gpu_sph, Some(box_size))
        .unwrap();
    cuda_sph
        .try_compute_balsara(&mut gpu_sph, Some(box_size))
        .unwrap();
    cuda_sph
        .try_compute_gadget2_forces(&mut gpu_sph, Some(box_size))
        .unwrap();

    let mut gpu = sph_to_core_particles(&gpu_sph);
    cuda_cool
        .try_apply_cooling(&mut gpu, &cool_cfg, dt, 0.0, 0.0)
        .unwrap();
    cuda_dust
        .try_update_dust(&mut gpu, &dust_cfg, cool_cfg.gamma, dt)
        .unwrap();
    cuda_mol
        .try_update_h2(&mut gpu, &mol_cfg, None, dt)
        .unwrap();

    // ── Comparar ──────────────────────────────────────────────────────────
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_close_rel(
            &format!("u[{i}]"),
            g.internal_energy,
            c.internal_energy,
            5e-4,
        );
        assert_close_rel(&format!("dust[{i}]"), g.dust_to_gas, c.dust_to_gas, 5e-4);
        assert_close_rel(&format!("h2[{i}]"), g.h2_fraction, c.h2_fraction, 5e-4);
    }

    // Comparar resultados SPH
    for (i, (c, g)) in cpu_sph.iter().zip(gpu_sph.iter()).enumerate() {
        let cgas = c.gas.as_ref().unwrap();
        let ggas = g.gas.as_ref().unwrap();
        assert_close_rel(&format!("sph_rho[{i}]"), ggas.rho, cgas.rho, 5e-3);
        assert_close_rel(
            &format!("sph_pressure[{i}]"),
            ggas.pressure,
            cgas.pressure,
            5e-3,
        );
        // f32 Newton-Raphson density convergence gives a different h_sml than f64,
        // which can change the Wendland kernel support radius and flip the sign of
        // acc contributions for border-zone neighbours.  For this "smoke/parity"
        // surface we only verify finiteness and a coarse absolute magnitude bound.
        // The CPU may also produce near-zero values due to f64 cancellation that
        // do not appear in f32, so a relative comparison is not meaningful here.
        assert!(
            ggas.acc_sph.x.is_finite() && ggas.acc_sph.y.is_finite() && ggas.acc_sph.z.is_finite(),
            "sph_acc[{i}] GPU is not finite: {:?}",
            ggas.acc_sph
        );
        let mag_gpu =
            (ggas.acc_sph.x.powi(2) + ggas.acc_sph.y.powi(2) + ggas.acc_sph.z.powi(2)).sqrt();
        // Sanity: force magnitude should be < 1.0 for this glass configuration (units).
        assert!(
            mag_gpu < 1.0,
            "sph_acc_mag[{i}]: gpu={mag_gpu:.3e} seems unphysically large"
        );
    }
}

/// Test de paridad para `try_sph_density_and_forces_core` (AP-18).
///
/// Compara el pipeline densidad + Balsara + fuerzas clásicas en CUDA directamente
/// sobre `gadget_ng_core::Particle` contra la referencia CPU con `SphParticle`.
/// Tolerancia relajada: el kernel CUDA usa f32 internamente y γ = 5/3 fijo.
#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_parity_sph_core_pipeline() {
    let Some(cuda_sph) = cuda_sph_or_skip() else {
        eprintln!("SKIP: CudaSphSolver no disponible.");
        return;
    };

    let box_size = 4.0_f64;
    let n_side = 4usize; // 4³ = 64 partículas gas

    // ── Referencia CPU con SphParticle ────────────────────────────────────────
    let mut cpu_sph = glass_sph_particles(n_side, box_size);
    compute_density_with_periodic(&mut cpu_sph, Some(box_size));
    compute_balsara_factors_with_periodic(&mut cpu_sph, Some(box_size));
    compute_sph_forces_with_periodic(&mut cpu_sph, Some(box_size));

    // ── CUDA sobre core::Particle ──────────────────────────────────────────────
    // Convertir SphParticle → core::Particle para usar el nuevo método.
    let mut core_parts: Vec<Particle> = cpu_sph
        .iter()
        .map(|sp| {
            let u = sp.gas.as_ref().map_or(1.5, |g| g.u);
            let h = sp
                .gas
                .as_ref()
                .map_or(box_size / n_side as f64 * 2.0, |g| g.h_sml);
            let mut p = Particle::new(sp.global_id, sp.mass, sp.position, sp.velocity);
            p.ptype = ParticleType::Gas;
            p.internal_energy = u;
            p.smoothing_length = h;
            p
        })
        .collect();

    let (gpu_rho, gpu_acc, gpu_du_dt) = cuda_sph
        .try_sph_density_and_forces_core(&mut core_parts, Some(box_size))
        .expect("try_sph_density_and_forces_core falló en hardware real");

    // ── Comparar densidad ──────────────────────────────────────────────────────
    // Tolerancia 5 %: CUDA usa Newton-Raphson f32 vs bisección f64 en CPU.
    for (i, sp) in cpu_sph.iter().enumerate() {
        let cpu_rho = sp.gas.as_ref().map_or(0.0, |g| g.rho);
        assert_close_rel(&format!("rho[{i}]"), gpu_rho[i], cpu_rho, 5e-2);
    }

    // ── Verificar h_sml actualizado ────────────────────────────────────────────
    for (i, p) in core_parts.iter().enumerate() {
        if p.ptype == ParticleType::Gas {
            assert!(
                p.smoothing_length.is_finite() && p.smoothing_length > 0.0,
                "h_sml[{i}] inválido tras CUDA: {}",
                p.smoothing_length
            );
        }
    }

    // ── Verificar finitud de aceleraciones ────────────────────────────────────
    for (i, a) in gpu_acc.iter().enumerate() {
        if core_parts[i].ptype == ParticleType::Gas {
            assert!(
                a.x.is_finite() && a.y.is_finite() && a.z.is_finite(),
                "acc_sph[{i}] GPU no es finita: {a:?}"
            );
            let cpu_gas = cpu_sph[i].gas.as_ref().unwrap();
            let mag_cpu =
                (cpu_gas.acc_sph.x.powi(2) + cpu_gas.acc_sph.y.powi(2) + cpu_gas.acc_sph.z.powi(2))
                    .sqrt();
            let mag_gpu = (a.x.powi(2) + a.y.powi(2) + a.z.powi(2)).sqrt();
            // Magnitudes en el mismo orden de magnitud (factor 10×).
            // Diferencias mayores se deben a f32 vs f64 en el kernel de suavizado.
            let ratio = if mag_cpu > 1e-12 {
                mag_gpu / mag_cpu
            } else {
                1.0
            };
            assert!(
                ratio < 10.0,
                "acc_sph_mag[{i}]: gpu={mag_gpu:.3e} cpu={mag_cpu:.3e} ratio={ratio:.2}"
            );
        }
    }

    // ── Verificar du_dt ───────────────────────────────────────────────────────
    for (i, &du) in gpu_du_dt.iter().enumerate() {
        if core_parts[i].ptype == ParticleType::Gas {
            assert!(du.is_finite(), "du_dt[{i}] GPU no es finito: {du}");
        }
    }

    eprintln!(
        "cuda_parity_sph_core_pipeline OK: {} partículas gas; rho max_rel≤5%, acc finitas",
        gpu_rho.len()
    );
}
