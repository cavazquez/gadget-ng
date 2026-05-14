use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use gadget_ng_core::{Particle, ParticleType, Vec3};
use gadget_ng_mhd::mean_gas_density;
use std::hint::black_box;

fn make_particles(n: usize) -> Vec<Particle> {
    (0..n)
        .map(|i| {
            let f = i as f64;
            let mut p = Particle::new_gas(
                i,
                1.0 + 0.001 * f,
                Vec3::new(0.01 * f, (0.03 * f).sin(), (0.05 * f).cos()),
                Vec3::zero(),
                1.0,
                0.12 + 0.001 * (i % 13) as f64,
            );
            if i % 17 == 0 {
                p.ptype = ParticleType::DarkMatter;
            }
            p
        })
        .collect()
}

fn mean_gas_density_scalar(particles: &[Particle]) -> f64 {
    let mut rho_sum = 0.0_f64;
    let mut n_gas = 0usize;
    for p in particles {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        let h = p.smoothing_length.max(1e-10);
        rho_sum += p.mass / (h * h * h);
        n_gas += 1;
    }
    if n_gas == 0 {
        1.0
    } else {
        rho_sum / n_gas as f64
    }
}

fn bench_mean_gas_density_scalar_vs_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("mhd_mean_gas_density_cpu");
    for n in [256usize, 4096, 16384] {
        let particles = make_particles(n);
        group.bench_with_input(
            BenchmarkId::new("scalar_reference", n),
            &particles,
            |b, parts| {
                b.iter(|| mean_gas_density_scalar(black_box(parts)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("simd_dispatch", n),
            &particles,
            |b, parts| {
                b.iter(|| mean_gas_density(black_box(parts)));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_mean_gas_density_scalar_vs_simd);
criterion_main!(benches);
