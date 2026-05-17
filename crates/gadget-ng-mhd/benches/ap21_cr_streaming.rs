//! AP-21 benchmark: CR streaming — scalar vs Rayon vs dispatch.
//!
//! Compares three execution backends for `streaming_crk`:
//! - `Scalar`   — serial O(N²) per-pair loop (pre-AP21 baseline)
//! - `Par`      — Rayon `par_iter` over particle pairs (AP-21)
//! - `Dispatch` — active dispatcher: routes to best path at compile time
//!
//! N sizes are intentionally small (O(N²) algorithm).
//!
//! Run with:
//! ```bash
//! cargo bench -p gadget-ng-mhd --bench ap21_cr_streaming --features bench-all-streaming-paths
//! ```
//! Results in `target/criterion/ap21_cr_streaming/`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_mhd::{StreamingBackend, streaming_crk_with_backend};
use std::hint::black_box;

fn make_cr_particles(n: usize) -> Vec<Particle> {
    (0..n)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            let mut p = Particle::new_gas(
                i,
                1.0,
                Vec3::new(0.5 + 0.4 * angle.cos(), 0.5 + 0.4 * angle.sin(), 0.0),
                Vec3::new(-angle.sin() * 0.1, angle.cos() * 0.1, 0.0),
                0.8,
                0.1,
            );
            p.cr_energy = 1e-4 * (1.0 + 0.1 * (i % 5) as f64);
            p.b_field = Vec3::new(1e-6 * (i % 3) as f64, 0.0, 0.0);
            p
        })
        .collect()
}

fn bench_cr_streaming_backends(c: &mut Criterion) {
    let dt = 1e5_f64;
    let coeff = 0.33_f64;

    let mut group = c.benchmark_group("ap21_cr_streaming");
    group.sample_size(20);

    for &n in &[32_usize, 64, 128, 256, 512] {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &n| {
            b.iter_batched(
                || make_cr_particles(n),
                |mut parts| {
                    streaming_crk_with_backend(
                        black_box(&mut parts),
                        black_box(dt),
                        black_box(coeff),
                        black_box(None),
                        StreamingBackend::Scalar,
                    );
                    parts
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("rayon_par", n), &n, |b, &n| {
            b.iter_batched(
                || make_cr_particles(n),
                |mut parts| {
                    streaming_crk_with_backend(
                        black_box(&mut parts),
                        black_box(dt),
                        black_box(coeff),
                        black_box(None),
                        StreamingBackend::Par,
                    );
                    parts
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("dispatch", n), &n, |b, &n| {
            b.iter_batched(
                || make_cr_particles(n),
                |mut parts| {
                    streaming_crk_with_backend(
                        black_box(&mut parts),
                        black_box(dt),
                        black_box(coeff),
                        black_box(None),
                        StreamingBackend::Dispatch,
                    );
                    parts
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_cr_streaming_backends);
criterion_main!(benches);
