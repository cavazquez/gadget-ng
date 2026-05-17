//! AP-21 benchmark: SPH forces Gadget-2 — scalar_ref (pre-AP21) vs batch_simd (post-AP21).
//!
//! Compares two implementations of `compute_sph_forces_gadget2`:
//! - `scalar_ref`  — pre-AP21 baseline: scalar `grad_w` per neighbour pair, no batching
//! - `batch_simd`  — post-AP21: collects neighbours, then calls `grad_w_batch` (AVX2/AVX-512)
//!
//! Both use the same outer Rayon `par_iter` loop; the difference is in the inner hot loop.
//!
//! Run with:
//! ```bash
//! cargo bench -p gadget-ng-sph --bench ap21_sph_forces --features bench-sph-forces-ref
//! ```
//! Results in `target/criterion/ap21_sph_forces/`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use gadget_ng_core::Vec3;
use gadget_ng_sph::{
    SphParticle, compute_sph_forces_gadget2, compute_sph_forces_gadget2_scalar_ref,
    density::compute_density,
};
use std::hint::black_box;

fn make_sph_particles(n: usize) -> Vec<SphParticle> {
    let n_side = (n as f64).cbrt().ceil() as usize;
    let n_actual = n_side.pow(3);
    let dx = 1.0 / n_side as f64;

    let mut parts: Vec<SphParticle> = (0..n_actual)
        .map(|k| {
            let iz = k / (n_side * n_side);
            let iy = (k / n_side) % n_side;
            let ix = k % n_side;
            let pos = Vec3::new(
                (ix as f64 + 0.5) * dx,
                (iy as f64 + 0.5) * dx,
                (iz as f64 + 0.5) * dx,
            );
            SphParticle::new_gas(k, 1.0, pos, Vec3::zero(), 1.0, 2.0 * dx)
        })
        .collect();

    compute_density(&mut parts);
    parts
}

fn bench_sph_forces(c: &mut Criterion) {
    let mut group = c.benchmark_group("ap21_sph_forces");
    group.sample_size(15);

    for &n in &[64_usize, 128, 256, 512, 1024] {
        let actual_n = {
            let n_side = (n as f64).cbrt().ceil() as usize;
            n_side.pow(3)
        };
        group.throughput(Throughput::Elements(actual_n as u64));

        group.bench_with_input(
            BenchmarkId::new("scalar_ref", actual_n),
            &actual_n,
            |b, &_n| {
                b.iter_batched(
                    || make_sph_particles(n),
                    |mut parts| {
                        compute_sph_forces_gadget2_scalar_ref(black_box(&mut parts));
                        parts
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("batch_simd", actual_n),
            &actual_n,
            |b, &_n| {
                b.iter_batched(
                    || make_sph_particles(n),
                    |mut parts| {
                        compute_sph_forces_gadget2(black_box(&mut parts));
                        parts
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_sph_forces);
criterion_main!(benches);
