//! AP-21 benchmark: Chemistry stiff solver — serial vs Rayon vs Rayon+SIMD.
//!
//! Compares three execution backends for `apply_chemistry`:
//! - `Serial`   — scalar per-particle (pre-AP21 fallback path)
//! - `Par`      — Rayon `par_iter_mut`, scalar `solve_chemistry_implicit` per particle
//! - `ParSimd`  — Rayon `par_chunks_mut(64)` + AVX2/AVX-512 slice dispatch (AP-21)
//!
//! Run with:
//! ```bash
//! cargo bench -p gadget-ng-rt --bench ap21_chemistry --features bench-all-chemistry-paths
//! ```
//! Results in `target/criterion/ap21_chemistry/`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_rt::{
    ChemParams, ChemState, ChemistryBackend, apply_chemistry_with_backend, m1::RadiationField,
};
use std::hint::black_box;

fn make_particles(n: usize) -> Vec<Particle> {
    (0..n)
        .map(|i| {
            Particle::new_gas(
                i,
                1.0,
                Vec3::new(
                    0.03 * i as f64,
                    0.02 * (i % 17) as f64,
                    0.01 * (i % 13) as f64,
                ),
                Vec3::zero(),
                0.7 + 0.04 * (i % 4) as f64,
                0.1,
            )
        })
        .collect()
}

fn make_chem_states(n: usize) -> Vec<ChemState> {
    (0..n)
        .map(|i| match i % 4 {
            0 => ChemState::neutral(),
            1 => ChemState::fully_ionized(),
            2 => {
                let mut st = ChemState::neutral();
                st.x_e = 1e-4;
                st
            }
            _ => {
                let mut st = ChemState::neutral();
                st.x_hii = 0.02;
                st.x_hi = 0.98;
                st.x_e = st.x_hii;
                st
            }
        })
        .collect()
}

fn bench_chemistry_backends(c: &mut Criterion) {
    let rad = RadiationField::uniform(8, 8, 8, 1.0, 1e-12);
    let params = ChemParams::default();
    let dt = 1e8;

    let mut group = c.benchmark_group("ap21_chemistry");
    group.sample_size(20);

    for &n in &[64_usize, 256, 1024, 4096] {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("serial", n), &n, |b, &n| {
            b.iter_batched(
                || (make_particles(n), make_chem_states(n)),
                |(mut parts, mut states)| {
                    apply_chemistry_with_backend(
                        black_box(&mut parts),
                        black_box(&mut states),
                        black_box(&rad),
                        black_box(&params),
                        black_box(dt),
                        ChemistryBackend::Serial,
                    );
                    (parts, states)
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("rayon_par", n), &n, |b, &n| {
            b.iter_batched(
                || (make_particles(n), make_chem_states(n)),
                |(mut parts, mut states)| {
                    apply_chemistry_with_backend(
                        black_box(&mut parts),
                        black_box(&mut states),
                        black_box(&rad),
                        black_box(&params),
                        black_box(dt),
                        ChemistryBackend::Par,
                    );
                    (parts, states)
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("rayon_par_simd", n), &n, |b, &n| {
            b.iter_batched(
                || (make_particles(n), make_chem_states(n)),
                |(mut parts, mut states)| {
                    apply_chemistry_with_backend(
                        black_box(&mut parts),
                        black_box(&mut states),
                        black_box(&rad),
                        black_box(&params),
                        black_box(dt),
                        ChemistryBackend::ParSimd,
                    );
                    (parts, states)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_chemistry_backends);
criterion_main!(benches);
