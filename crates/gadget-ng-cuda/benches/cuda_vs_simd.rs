use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use gadget_ng_core::{DirectGravity, GravitySolver, Particle, ParticleType, SimdDirectGravity, Vec3};
use gadget_ng_cuda::{CudaDirectGravity, CudaMhdSolver, CudaRtSolver, CudaSphSolver, CudaTreeSolver};
use gadget_ng_mhd::apply_flux_freeze;
use gadget_ng_pm::PmSolver;
use gadget_ng_rt::{M1Params, RadiationField, m1_update};
use gadget_ng_sph::{SphParticle, compute_density};
use gadget_ng_tree::{RemoteMultipoleNode, RmnSoa};
use std::hint::black_box;

// ── Helpers comunes ──────────────────────────────────────────────────────────

fn make_positions(n: usize) -> (Vec<Vec3>, Vec<f64>, Vec<[f32; 3]>, Vec<f32>) {
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
    (positions64, masses64, positions32, masses32)
}

fn make_particles_core(n: usize) -> Vec<Particle> {
    (0..n)
        .map(|i| {
            let t = i as f64 * 0.37;
            let mut p = Particle::new(
                i,
                1.0 / n as f64,
                Vec3::new(t.sin(), t.cos(), (t * 0.5).sin()),
                Vec3::new(0.1 * t.cos(), 0.1 * t.sin(), 0.0),
            );
            if i % 3 == 0 {
                p.ptype = ParticleType::Gas;
                p.internal_energy = 0.5;
                p.smoothing_length = 0.1;
            }
            p
        })
        .collect()
}

fn make_sph_particles(n: usize) -> Vec<SphParticle> {
    (0..n)
        .map(|i| {
            let t = i as f64 * 0.37;
            SphParticle::new_gas(
                i,
                1.0 / n as f64,
                Vec3::new(t.sin(), t.cos(), (t * 0.5).sin()),
                Vec3::new(0.1 * t.cos(), 0.1 * t.sin(), 0.0),
                0.5,
                0.1,
            )
        })
        .collect()
}

// ── Benchmark: Gravedad directa ──────────────────────────────────────────────

fn bench_direct_cuda_vs_simd(c: &mut Criterion) {
    let cuda = CudaDirectGravity::try_new_checked(0.1).ok();
    if cuda.is_none() {
        eprintln!("cuda_vs_simd: CUDA no disponible; se omite la serie cuda");
    }

    let mut group = c.benchmark_group("cuda_vs_simd_direct");
    group.sample_size(10);

    for n in [512usize, 1024, 2048] {
        let (positions64, masses64, positions32, masses32) = make_positions(n);
        let indices: Vec<usize> = (0..n).collect();
        let eps2 = 0.01_f64;
        let g = 1.0_f64;
        let mut out = vec![Vec3::zero(); n];

        group.bench_with_input(BenchmarkId::new("cpu_serial", n), &n, |b, _| {
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

        group.bench_with_input(BenchmarkId::new("cpu_simd", n), &n, |b, _| {
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

        if let Some(ref cuda) = cuda {
            group.bench_with_input(BenchmarkId::new("cuda", n), &n, |b, _| {
                b.iter(|| {
                    black_box(cuda.compute(black_box(&positions32), black_box(&masses32)));
                });
            });
        }
    }

    group.finish();
}

// ── Benchmark: PM (CIC + FFT + Poisson) ────────────────────────────────────

fn bench_pm_cuda_vs_cpu(c: &mut Criterion) {
    use gadget_ng_cuda::CudaPmSolver;

    let mut group = c.benchmark_group("cuda_vs_simd_pm");
    group.sample_size(10);

    for grid in [32usize, 64] {
        let n = grid * grid;
        let (positions64, masses64, _, _) = make_positions(n);
        let indices: Vec<usize> = (0..n).collect();
        let mut out = vec![Vec3::zero(); n];

        // CPU PM solver (usa GravitySolver trait)
        let cpu_pm = PmSolver::new(grid, 20.0);
        group.bench_with_input(BenchmarkId::new("cpu_pm", grid), &grid, |b, _| {
            b.iter(|| {
                cpu_pm.accelerations_for_indices(
                    black_box(&positions64),
                    black_box(&masses64),
                    0.0,
                    1.0,
                    black_box(&indices),
                    black_box(&mut out),
                );
            });
        });

        // CUDA PM solver
        let cuda_pm = CudaPmSolver::try_new(grid, 20.0);
        if let Some(ref cuda) = cuda_pm {
            let mut cuda_out = vec![Vec3::zero(); n];
            group.bench_with_input(BenchmarkId::new("cuda_pm", grid), &grid, |b, _| {
                b.iter(|| {
                    let _ = black_box(cuda.try_accelerations_for_indices(
                        black_box(&positions64),
                        black_box(&masses64),
                        0.0,
                        1.0,
                        black_box(&indices),
                        black_box(&mut cuda_out),
                    ));
                });
            });
        }
    }

    group.finish();
}

// ── Benchmark: SPH density ──────────────────────────────────────────────────

fn bench_sph_cuda_vs_cpu(c: &mut Criterion) {
    let cuda_sph = CudaSphSolver::try_new_checked().ok();
    if cuda_sph.is_none() {
        eprintln!("bench_sph: CUDA SPH no disponible; solo serie cpu");
    }

    let mut group = c.benchmark_group("cuda_vs_simd_sph");
    group.sample_size(10);

    for n in [512usize, 1024] {
        let template = make_sph_particles(n);

        // CPU density
        group.bench_with_input(BenchmarkId::new("cpu_sph_density", n), &n, |b, _| {
            b.iter(|| {
                let mut local = template.clone();
                black_box(compute_density(black_box(&mut local)));
            });
        });

        // CUDA SPH density
        if let Some(ref cuda) = cuda_sph {
            group.bench_with_input(BenchmarkId::new("cuda_sph_density", n), &n, |b, _| {
                b.iter(|| {
                    let mut local = template.clone();
                    let _ = black_box(cuda.try_compute_density(black_box(&mut local), None));
                });
            });
        }
    }

    group.finish();
}

// ── Benchmark: MHD flux_freeze ───────────────────────────────────────────────

fn bench_mhd_cuda_vs_cpu(c: &mut Criterion) {
    let cuda_mhd = CudaMhdSolver::try_new_checked().ok();
    if cuda_mhd.is_none() {
        eprintln!("bench_mhd: CUDA MHD no disponible; solo serie cpu");
    }

    let mut group = c.benchmark_group("cuda_vs_simd_mhd");
    group.sample_size(10);

    for n in [1024usize, 4096] {
        let template = make_particles_core(n);

        // CPU MHD flux-freeze
        group.bench_with_input(BenchmarkId::new("cpu_mhd_flux_freeze", n), &n, |b, _| {
            b.iter(|| {
                let mut local = template.clone();
                black_box(apply_flux_freeze(black_box(&mut local), 5.0 / 3.0, 1.0e-6, 1.0));
            });
        });

        // CUDA MHD flux-freeze
        if let Some(ref cuda) = cuda_mhd {
            group.bench_with_input(BenchmarkId::new("cuda_mhd_flux_freeze", n), &n, |b, _| {
                b.iter(|| {
                    let mut local = template.clone();
                    let _ = black_box(cuda.try_apply_flux_freeze(
                        black_box(&mut local),
                        5.0 / 3.0,
                        1.0e-6,
                        1.0,
                    ));
                });
            });
        }
    }

    group.finish();
}

// ── Benchmark: RT M1 advección ───────────────────────────────────────────────

fn bench_rt_cuda_vs_cpu(c: &mut Criterion) {
    let cuda_rt = CudaRtSolver::try_new_checked().ok();
    if cuda_rt.is_none() {
        eprintln!("bench_rt: CUDA RT no disponible; solo serie cpu");
    }

    let mut group = c.benchmark_group("cuda_vs_simd_rt");
    group.sample_size(10);

    // Grilla pequeña (8³=512 celdas) para que 1 paso CPU sea ~ms, no segundos.
    for grid in [8usize, 16] {
        let dx = 1.0 / grid as f64;
        let params = M1Params {
            c_red_factor: 100.0,
            kappa_abs: 1.0,
            substeps: 1,
            ..Default::default()
        };
        // dt = CFL con c_red = 1 (unidades internas)
        let dt = 0.3 * dx;

        // CPU RT M1
        group.bench_with_input(BenchmarkId::new("cpu_rt_m1", grid), &grid, |b, _| {
            b.iter(|| {
                let mut rad = RadiationField::uniform(grid, grid, grid, dx, 1.0e-4);
                black_box(m1_update(black_box(&mut rad), dt, black_box(&params)));
            });
        });

        // CUDA RT M1
        if let Some(ref cuda) = cuda_rt {
            group.bench_with_input(BenchmarkId::new("cuda_rt_m1", grid), &grid, |b, _| {
                b.iter(|| {
                    let mut rad = RadiationField::uniform(grid, grid, grid, dx, 1.0e-4);
                    let _ = black_box(cuda.try_m1_advection(
                        black_box(&mut rad),
                        dt,
                        black_box(&params),
                    ));
                });
            });
        }
    }

    group.finish();
}

// ── Benchmark: Tree LET accel ────────────────────────────────────────────────

fn bench_tree_cuda_vs_cpu(c: &mut Criterion) {
    let cuda_tree = CudaTreeSolver::try_new_checked().ok();
    if cuda_tree.is_none() {
        eprintln!("bench_tree: CUDA Tree no disponible; solo serie cpu");
    }

    let mut group = c.benchmark_group("cuda_vs_simd_tree");
    group.sample_size(10);

    for n_nodes in [256usize, 1024, 4096] {
        let n_particles = n_nodes / 4;
        let particles: Vec<Particle> = (0..n_particles)
            .map(|i| {
                let t = i as f64 * 0.37;
                Particle::new(
                    i,
                    1.0 / n_particles as f64,
                    Vec3::new(0.3 * t.sin(), 0.3 * t.cos(), 0.3 * (t * 0.5).sin()),
                    Vec3::zero(),
                )
            })
            .collect();

        let rmns: Vec<RemoteMultipoleNode> = (0..n_nodes)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / n_nodes as f64;
                RemoteMultipoleNode {
                    com: Vec3::new(3.0 * angle.cos(), 3.0 * angle.sin(), 0.0),
                    mass: 1.0 / n_nodes as f64,
                    quad: [0.0; 6],
                    oct: [0.0; 7],
                    hex: [0.0; 15],
                    half_size: 0.1,
                }
            })
            .collect();
        let nodes = RmnSoa::from_slice(&rmns);

        // CPU LET accel
        group.bench_with_input(BenchmarkId::new("cpu_let", n_nodes), &n_nodes, |b, _| {
            b.iter(|| {
                for p in &particles {
                    let _ = black_box(nodes.accel(black_box(p.position), 1.0, 1.0e-4));
                }
            });
        });

        // CUDA LET
        if let Some(ref cuda) = cuda_tree {
            group.bench_with_input(BenchmarkId::new("cuda_let", n_nodes), &n_nodes, |b, _| {
                b.iter(|| {
                    let _ = black_box(cuda.try_tree_walk_let(
                        black_box(&particles),
                        black_box(&nodes),
                        1.0,
                        1.0e-4,
                    ));
                });
            });
        }
    }

    group.finish();
}

// ── Registro de grupos ────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_direct_cuda_vs_simd,
    bench_pm_cuda_vs_cpu,
    bench_sph_cuda_vs_cpu,
    bench_mhd_cuda_vs_cpu,
    bench_rt_cuda_vs_cpu,
    bench_tree_cuda_vs_cpu,
);
criterion_main!(benches);
