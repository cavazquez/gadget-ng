//! Compara `dedner_cleaning_step` en cuatro backends CPU (Criterion).
//!
//! Requiere `feature = "bench-all-dedner-paths"` (ver `Cargo.toml`). Tras medir:
//!
//! ```bash
//! cargo bench -p gadget-ng-mhd --bench dedner_backend_bench --features bench-all-dedner-paths
//! python3 scripts/plot_dedner_backend_benchmark.py
//! ```

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_mhd::{DednerCleaningBackend, dedner_cleaning_step_with_backend};
use std::f64::consts::TAU;
use std::hint::black_box;

fn make_particles(n: usize) -> Vec<Particle> {
    (0..n)
        .map(|i| {
            let x = (i as f64) / (n as f64).max(1.0);
            let mut p = Particle::new_gas(
                i,
                1.0,
                Vec3::new(x, 0.0, 0.0),
                Vec3::new(0.01, 0.0, 0.0),
                1.0,
                0.15,
            );
            p.b_field = Vec3::new(1.0, 0.05 * (x * TAU).sin(), 0.0);
            p.psi_div = 0.0;
            p
        })
        .collect()
}

fn bench_dedner_cleaning_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("dedner_cleaning_backends");
    group.sample_size(40);
    let c_h = 1.0_f64;
    let c_r = 0.5_f64;
    let dt = 0.001_f64;

    for n in [256_usize, 1024] {
        group.bench_with_input(BenchmarkId::new("cpu_sin_rayon_scalar", n), &n, |b, &n| {
            let particles = make_particles(n);
            b.iter_batched(
                || particles.clone(),
                |mut p| {
                    dedner_cleaning_step_with_backend(
                        black_box(&mut p),
                        c_h,
                        c_r,
                        dt,
                        DednerCleaningBackend::CpuSinRayonScalar,
                    );
                    p
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("cpu_con_rayon", n), &n, |b, &n| {
            let particles = make_particles(n);
            b.iter_batched(
                || particles.clone(),
                |mut p| {
                    dedner_cleaning_step_with_backend(
                        black_box(&mut p),
                        c_h,
                        c_r,
                        dt,
                        DednerCleaningBackend::CpuRayon,
                    );
                    p
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("simd_sin_rayon_avx2", n), &n, |b, &n| {
            let particles = make_particles(n);
            b.iter_batched(
                || particles.clone(),
                |mut p| {
                    dedner_cleaning_step_with_backend(
                        black_box(&mut p),
                        c_h,
                        c_r,
                        dt,
                        DednerCleaningBackend::SimdSinRayonAvx2,
                    );
                    p
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("simd_sin_rayon_avx512", n), &n, |b, &n| {
            let particles = make_particles(n);
            b.iter_batched(
                || particles.clone(),
                |mut p| {
                    dedner_cleaning_step_with_backend(
                        black_box(&mut p),
                        c_h,
                        c_r,
                        dt,
                        DednerCleaningBackend::SimdSinRayonAvx512,
                    );
                    p
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_dedner_cleaning_backends);
criterion_main!(benches);
