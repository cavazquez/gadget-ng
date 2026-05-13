use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use gadget_ng_core::{
    CoolingKind, DirectGravity, DustSection, GravitySolver, MolecularSection, Particle, SphSection,
    Vec3,
};
use gadget_ng_cuda::{
    CudaCoolingSolver, CudaDirectGravity, CudaDustSolver, CudaMhdSolver, CudaMolecularSolver,
    CudaRtSolver, CudaSphSolver,
};
use gadget_ng_mhd::apply_flux_freeze;
use gadget_ng_mhd::b_field_stats;
use gadget_ng_rt::{M1Params, RadiationField, radiation_gas_coupling_step};
use gadget_ng_sph::{
    apply_cooling_with_redshift, compute_balsara_factors_with_periodic,
    compute_density_with_periodic, compute_sph_forces_gadget2_with_periodic,
    compute_sph_forces_with_periodic, dust::update_dust, molecular_gas::update_h2_fraction,
    particle::SphParticle,
};
use std::hint::black_box;

// ── Data generation ───────────────────────────────────────────────────────────

fn make_sph_glass(n_side: usize, box_size: f64) -> Vec<SphParticle> {
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

fn make_core_gas(n: usize, box_size: f64) -> Vec<Particle> {
    let mut parts = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 * 0.37;
        let x = (t.sin() * 0.8 + 0.5) * box_size;
        let y = (t.cos() * 0.8 + 0.5) * box_size;
        let z = ((t * 0.5).sin() * 0.8 + 0.5) * box_size;
        let mut p = Particle::new_gas(i, 1.0, Vec3::new(x, y, z), Vec3::zero(), 1.0, 0.5);
        p.internal_energy = 1.5;
        p.dust_to_gas = 0.002;
        p.h2_fraction = 0.05;
        p.metallicity = 0.02;
        p.b_field = Vec3::new(0.1, -0.2, 0.15);
        p.smoothing_length = 0.5;
        parts.push(p);
    }
    parts
}

fn make_direct_gravity_data(
    n: usize,
) -> (Vec<Vec3>, Vec<f64>, Vec<[f32; 3]>, Vec<f32>, Vec<usize>) {
    let positions64: Vec<Vec3> = (0..n)
        .map(|i| {
            let t = i as f64 * 0.37;
            Vec3::new(t.sin() * 10.0, t.cos() * 10.0, (t * 0.5).sin() * 10.0)
        })
        .collect();
    let masses64 = vec![1.0_f64 / n as f64; n];
    let positions32 = positions64
        .iter()
        .map(|p| [p.x as f32, p.y as f32, p.z as f32])
        .collect();
    let masses32 = masses64.iter().map(|&m| m as f32).collect();
    let indices: Vec<usize> = (0..n).collect();
    (positions64, masses64, positions32, masses32, indices)
}

fn cooling_cfg() -> SphSection {
    let mut c = SphSection::default();
    c.gamma = 5.0 / 3.0;
    c.t_floor_k = 1e4;
    c.cooling = CoolingKind::AtomicHHe;
    c
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

fn mol_cfg() -> MolecularSection {
    MolecularSection {
        enabled: true,
        rho_h2_threshold: 10.0,
        ..Default::default()
    }
}

fn rt_field() -> RadiationField {
    RadiationField::uniform(8, 8, 8, 0.5, 1e-5)
}

fn rt_params() -> M1Params {
    M1Params {
        c_red_factor: 100.0,
        kappa_abs: 1e-2,
        kappa_scat: 0.0,
        substeps: 1,
        sigma_dust: 0.1,
    }
}

// ── Benchmark 1: SPH density ──────────────────────────────────────────────────

fn bench_sph_density(c: &mut Criterion) {
    let cuda = CudaSphSolver::try_new_checked().ok();
    let mut group = c.benchmark_group("parity_sph_density");
    group.sample_size(10);
    let box_size = 4.0_f64;

    let sides = [4usize, 6, 8]; // 64, 216, 512 particles

    for &n_side in &sides {
        let n = n_side.pow(3);
        let mut cpu = make_sph_glass(n_side, box_size);
        let mut gpu = cpu.clone();

        group.bench_with_input(BenchmarkId::new("cpu_rayon", n), &n, |b, _| {
            b.iter(|| {
                compute_density_with_periodic(black_box(&mut cpu), Some(box_size));
            });
        });

        if let Some(cuda) = cuda.as_ref() {
            group.bench_with_input(BenchmarkId::new("cuda", n), &n, |b, _| {
                b.iter(|| {
                    cuda.try_compute_density(black_box(&mut gpu), Some(box_size))
                        .unwrap();
                });
            });
        }
    }
    group.finish();
}

// ── Benchmark 2: SPH Balsara ──────────────────────────────────────────────────

fn bench_sph_balsara(c: &mut Criterion) {
    let cuda = CudaSphSolver::try_new_checked().ok();
    let mut group = c.benchmark_group("parity_sph_balsara");
    group.sample_size(10);
    let box_size = 4.0_f64;

    for &n_side in &[4usize, 6, 8] {
        let n = n_side.pow(3);
        // Precompute density (shared by CPU and CUDA)
        let mut cpu_dens = make_sph_glass(n_side, box_size);
        compute_density_with_periodic(&mut cpu_dens, Some(box_size));
        let mut cpu = cpu_dens.clone();
        let mut gpu = cpu_dens.clone();

        group.bench_with_input(BenchmarkId::new("cpu_rayon", n), &n, |b, _| {
            b.iter(|| {
                compute_balsara_factors_with_periodic(black_box(&mut cpu), Some(box_size));
            });
        });

        if let Some(cuda) = cuda.as_ref() {
            group.bench_with_input(BenchmarkId::new("cuda", n), &n, |b, _| {
                b.iter(|| {
                    cuda.try_compute_balsara(black_box(&mut gpu), Some(box_size))
                        .unwrap();
                });
            });
        }
    }
    group.finish();
}

// ── Benchmark 3: SPH forces (classical Monaghan) ──────────────────────────────

fn bench_sph_forces(c: &mut Criterion) {
    let cuda = CudaSphSolver::try_new_checked().ok();
    let mut group = c.benchmark_group("parity_sph_forces");
    group.sample_size(10);
    let box_size = 4.0_f64;

    for &n_side in &[4usize, 6, 8] {
        let n = n_side.pow(3);
        let mut cpu_dens = make_sph_glass(n_side, box_size);
        compute_density_with_periodic(&mut cpu_dens, Some(box_size));
        let mut cpu = cpu_dens.clone();
        let mut gpu = cpu_dens.clone();

        group.bench_with_input(BenchmarkId::new("cpu_rayon", n), &n, |b, _| {
            b.iter(|| {
                compute_sph_forces_with_periodic(black_box(&mut cpu), Some(box_size));
            });
        });

        if let Some(cuda) = cuda.as_ref() {
            group.bench_with_input(BenchmarkId::new("cuda", n), &n, |b, _| {
                b.iter(|| {
                    cuda.try_compute_forces(black_box(&mut gpu), Some(box_size))
                        .unwrap();
                });
            });
        }
    }
    group.finish();
}

// ── Benchmark 4: SPH Gadget-2 forces (signal velocity + entropy) ──────────────

fn bench_sph_gadget2(c: &mut Criterion) {
    let cuda = CudaSphSolver::try_new_checked().ok();
    let mut group = c.benchmark_group("parity_sph_gadget2");
    group.sample_size(10);
    let box_size = 4.0_f64;

    for &n_side in &[4usize, 6, 8] {
        let n = n_side.pow(3);
        let mut cpu_dens = make_sph_glass(n_side, box_size);
        compute_density_with_periodic(&mut cpu_dens, Some(box_size));
        compute_balsara_factors_with_periodic(&mut cpu_dens, Some(box_size));
        let mut cpu = cpu_dens.clone();
        let mut gpu = cpu_dens.clone();

        group.bench_with_input(BenchmarkId::new("cpu_rayon", n), &n, |b, _| {
            b.iter(|| {
                compute_sph_forces_gadget2_with_periodic(black_box(&mut cpu), Some(box_size));
            });
        });

        if let Some(cuda) = cuda.as_ref() {
            group.bench_with_input(BenchmarkId::new("cuda", n), &n, |b, _| {
                b.iter(|| {
                    cuda.try_compute_gadget2_forces(black_box(&mut gpu), Some(box_size))
                        .unwrap();
                });
            });
        }
    }
    group.finish();
}

// ── Benchmark 5: Cooling (AtomicHHe) ──────────────────────────────────────────

fn bench_cooling(c: &mut Criterion) {
    let cuda = CudaCoolingSolver::try_new_checked().ok();
    let mut group = c.benchmark_group("parity_cooling");
    group.sample_size(20);
    let cfg = cooling_cfg();
    let box_size = 4.0_f64;

    for &n in &[64usize, 256, 1024, 4096] {
        let mut cpu = make_core_gas(n, box_size);
        let mut gpu = cpu.clone();

        group.bench_with_input(BenchmarkId::new("cpu_rayon", n), &n, |b, _| {
            b.iter(|| {
                apply_cooling_with_redshift(black_box(&mut cpu), &cfg, 0.01, 0.0);
            });
        });

        if let Some(cuda) = cuda.as_ref() {
            group.bench_with_input(BenchmarkId::new("cuda", n), &n, |b, _| {
                b.iter(|| {
                    cuda.try_apply_cooling(black_box(&mut gpu), &cfg, 0.01, 0.0, 0.0)
                        .unwrap();
                });
            });
        }
    }
    group.finish();
}

// ── Benchmark 6: Dust evolution ───────────────────────────────────────────────

fn bench_dust(c: &mut Criterion) {
    let cuda = CudaDustSolver::try_new_checked().ok();
    let mut group = c.benchmark_group("parity_dust");
    group.sample_size(20);
    let dust_cfg = dust_cfg();
    let box_size = 4.0_f64;

    for &n in &[64usize, 256, 1024, 4096] {
        let mut cpu = make_core_gas(n, box_size);
        let mut gpu = cpu.clone();

        group.bench_with_input(BenchmarkId::new("cpu_rayon", n), &n, |b, _| {
            b.iter(|| {
                update_dust(black_box(&mut cpu), &dust_cfg, 5.0 / 3.0, 0.1);
            });
        });

        if let Some(cuda) = cuda.as_ref() {
            group.bench_with_input(BenchmarkId::new("cuda", n), &n, |b, _| {
                b.iter(|| {
                    cuda.try_update_dust(black_box(&mut gpu), &dust_cfg, 5.0 / 3.0, 0.1)
                        .unwrap();
                });
            });
        }
    }
    group.finish();
}

// ── Benchmark 7: H2 fraction ──────────────────────────────────────────────────

fn bench_h2(c: &mut Criterion) {
    let cuda = CudaMolecularSolver::try_new_checked().ok();
    let mut group = c.benchmark_group("parity_h2");
    group.sample_size(20);
    let mol_cfg = mol_cfg();
    let box_size = 4.0_f64;

    for &n in &[64usize, 256, 1024, 4096] {
        let mut cpu = make_core_gas(n, box_size);
        let mut gpu = cpu.clone();

        group.bench_with_input(BenchmarkId::new("cpu_rayon", n), &n, |b, _| {
            b.iter(|| {
                update_h2_fraction(black_box(&mut cpu), &mol_cfg, 0.05);
            });
        });

        if let Some(cuda) = cuda.as_ref() {
            group.bench_with_input(BenchmarkId::new("cuda", n), &n, |b, _| {
                b.iter(|| {
                    cuda.try_update_h2(black_box(&mut gpu), &mol_cfg, None, 0.05)
                        .unwrap();
                });
            });
        }
    }
    group.finish();
}

// ── Benchmark 8: MHD flux freeze ──────────────────────────────────────────────

fn bench_mhd_flux_freeze(c: &mut Criterion) {
    let cuda = CudaMhdSolver::try_new_checked().ok();
    let mut group = c.benchmark_group("parity_mhd_flux_freeze");
    group.sample_size(20);
    let gamma = 5.0 / 3.0;
    let beta_freeze = 100.0;
    let box_size = 4.0_f64;

    for &n in &[64usize, 256, 1024, 4096] {
        let mut cpu = make_core_gas(n, box_size);
        let rho_ref = cpu
            .iter()
            .filter(|p| p.ptype == gadget_ng_core::ParticleType::Gas)
            .map(|p| {
                let h = p.smoothing_length.max(1e-10);
                p.mass / (h * h * h)
            })
            .sum::<f64>()
            / n as f64;
        let mut gpu = cpu.clone();

        group.bench_with_input(BenchmarkId::new("cpu_rayon", n), &n, |b, _| {
            b.iter(|| {
                apply_flux_freeze(
                    black_box(&mut cpu),
                    black_box(gamma),
                    black_box(beta_freeze),
                    black_box(rho_ref),
                );
            });
        });

        if let Some(cuda) = cuda.as_ref() {
            group.bench_with_input(BenchmarkId::new("cuda", n), &n, |b, _| {
                b.iter(|| {
                    cuda.try_apply_flux_freeze(black_box(&mut gpu), gamma, beta_freeze, rho_ref)
                        .unwrap();
                });
            });
        }
    }
    group.finish();
}

// ── Benchmark 9: MHD B-field stats ────────────────────────────────────────────

fn bench_mhd_b_stats(c: &mut Criterion) {
    let cuda = CudaMhdSolver::try_new_checked().ok();
    let mut group = c.benchmark_group("parity_mhd_b_stats");
    group.sample_size(20);
    let box_size = 4.0_f64;

    for &n in &[64usize, 256, 1024, 4096] {
        let cpu = make_core_gas(n, box_size);
        let gpu = cpu.clone();

        group.bench_with_input(BenchmarkId::new("cpu_rayon", n), &n, |b, _| {
            b.iter(|| {
                black_box(b_field_stats(black_box(&cpu)));
            });
        });

        if let Some(cuda) = cuda.as_ref() {
            group.bench_with_input(BenchmarkId::new("cuda", n), &n, |b, _| {
                b.iter(|| {
                    black_box(cuda.try_b_field_stats(black_box(&gpu)).unwrap());
                });
            });
        }
    }
    group.finish();
}

// ── Benchmark 10: RT photoheating ─────────────────────────────────────────────

fn bench_rt_photoheating(c: &mut Criterion) {
    let cuda = CudaRtSolver::try_new_checked().ok();
    let mut group = c.benchmark_group("parity_rt_photoheating");
    group.sample_size(10);
    let box_size = 4.0_f64;
    let m1p = rt_params();

    for &n in &[64usize, 256, 1024] {
        let mut cpu = make_core_gas(n, box_size);
        let mut gpu = cpu.clone();
        let mut rad = rt_field();
        let gamma_hi: Vec<f64> = vec![1e-12_f64; rad.n_cells()];

        group.bench_with_input(BenchmarkId::new("cpu_rayon", n), &n, |b, _| {
            b.iter(|| {
                radiation_gas_coupling_step(
                    black_box(&mut cpu),
                    black_box(&mut rad),
                    &m1p,
                    black_box(0.01),
                    black_box(box_size),
                );
            });
        });

        if let Some(cuda) = cuda.as_ref() {
            group.bench_with_input(BenchmarkId::new("cuda", n), &n, |b, _| {
                b.iter(|| {
                    cuda.try_apply_photoheating(
                        black_box(&mut gpu),
                        &rad,
                        &gamma_hi,
                        0.01,
                        box_size,
                    )
                    .unwrap();
                });
            });
        }
    }
    group.finish();
}

// ── Benchmark 11: Direct gravity ──────────────────────────────────────────────

fn bench_direct_gravity(c: &mut Criterion) {
    let cuda = CudaDirectGravity::try_new_checked(0.1).ok();
    let mut group = c.benchmark_group("parity_direct_gravity");
    group.sample_size(10);

    for &n in &[512usize, 1024, 2048] {
        let (positions64, masses64, positions32, masses32, indices) = make_direct_gravity_data(n);
        let eps2 = 0.01_f64;
        let g = 1.0_f64;

        group.bench_with_input(BenchmarkId::new("cpu_serial", n), &n, |b, _| {
            let mut out = vec![Vec3::zero(); n];
            b.iter(|| {
                DirectGravity.accelerations_for_indices(
                    black_box(&positions64),
                    black_box(&masses64),
                    black_box(eps2),
                    black_box(g),
                    black_box(&indices),
                    black_box(&mut out),
                );
            });
        });

        #[cfg(feature = "simd")]
        {
            use gadget_ng_core::{RayonDirectGravity, SimdDirectGravity};
            group.bench_with_input(BenchmarkId::new("cpu_rayon", n), &n, |b, _| {
                let mut out = vec![Vec3::zero(); n];
                b.iter(|| {
                    RayonDirectGravity.accelerations_for_indices(
                        black_box(&positions64),
                        black_box(&masses64),
                        black_box(eps2),
                        black_box(g),
                        black_box(&indices),
                        black_box(&mut out),
                    );
                });
            });
            group.bench_with_input(BenchmarkId::new("cpu_simd", n), &n, |b, _| {
                let mut out = vec![Vec3::zero(); n];
                b.iter(|| {
                    SimdDirectGravity.accelerations_for_indices(
                        black_box(&positions64),
                        black_box(&masses64),
                        black_box(eps2),
                        black_box(g),
                        black_box(&indices),
                        black_box(&mut out),
                    );
                });
            });
        }

        if let Some(cuda) = cuda.as_ref() {
            group.bench_with_input(BenchmarkId::new("cuda", n), &n, |b, _| {
                b.iter(|| {
                    black_box(cuda.compute(black_box(&positions32), black_box(&masses32)));
                });
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sph_density,
    bench_sph_balsara,
    bench_sph_forces,
    bench_sph_gadget2,
    bench_cooling,
    bench_dust,
    bench_h2,
    bench_mhd_flux_freeze,
    bench_mhd_b_stats,
    bench_rt_photoheating,
    bench_direct_gravity,
);
criterion_main!(benches);
