//! Benchmark MHD — Phase 132: medir `advance_induction`, `apply_magnetic_forces`
//! y `dedner_cleaning_step` sobre N = 100, 500, 1000 partículas.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_mhd::{advance_induction, apply_magnetic_forces, dedner_cleaning_step};

fn make_particles(n: usize) -> Vec<Particle> {
    (0..n).map(|i| {
        let x = (i as f64) / (n as f64);
        let mut p = Particle::new_gas(i, 1.0, Vec3::new(x, 0.0, 0.0), Vec3::new(0.01, 0.0, 0.0), 1.0, 0.15);
        p.b_field = Vec3::new(1.0, 0.05 * (x * 6.28).sin(), 0.0);
        p.psi_div = 0.0;
        p
    }).collect()
}

fn bench_mhd_stack(c: &mut Criterion) {
    let mut group = c.benchmark_group("mhd_stack");
    for &n in &[100_usize, 500, 1000] {
        group.bench_with_input(BenchmarkId::new("advance_induction", n), &n, |b, &n| {
            let mut particles = make_particles(n);
            b.iter(|| {
                advance_induction(&mut particles, 0.001);
            });
        });

        group.bench_with_input(BenchmarkId::new("apply_magnetic_forces", n), &n, |b, &n| {
            let mut particles = make_particles(n);
            b.iter(|| {
                apply_magnetic_forces(&mut particles, 0.001);
            });
        });

        group.bench_with_input(BenchmarkId::new("dedner_cleaning", n), &n, |b, &n| {
            let mut particles = make_particles(n);
            b.iter(|| {
                dedner_cleaning_step(&mut particles, 1.0, 0.5, 0.001);
            });
        });

        group.bench_with_input(BenchmarkId::new("full_mhd_step", n), &n, |b, &n| {
            let mut particles = make_particles(n);
            b.iter(|| {
                let dt = 0.001;
                advance_induction(&mut particles, dt);
                apply_magnetic_forces(&mut particles, dt);
                dedner_cleaning_step(&mut particles, 1.0, 0.5, dt);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_mhd_stack);
criterion_main!(benches);
