//! Benchmarks de feedback AGN: `apply_agn_feedback` y `bondi_accretion_rate`.
//!
//! Mide el overhead en función del número de partículas de gas N y
//! el número de agujeros negros n_bh.
//!
//! Ejecutar con:
//! ```bash
//! cargo bench -p gadget-ng-sph --bench agn_feedback
//! ```
//! Resultados en `target/criterion/agn_feedback/`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_sph::{
    AgnParams, BlackHole, apply_agn_feedback, bondi_accretion_rate, grow_black_holes,
};

fn make_gas_particles(n: usize, box_size: f64) -> Vec<Particle> {
    let n_side = (n as f64).cbrt().ceil() as usize;
    let dx = box_size / n_side as f64;
    let mut parts = Vec::with_capacity(n);
    'outer: for ix in 0..n_side {
        for iy in 0..n_side {
            for iz in 0..n_side {
                let id = ix * n_side * n_side + iy * n_side + iz;
                let pos = Vec3::new(
                    (ix as f64 + 0.5) * dx,
                    (iy as f64 + 0.5) * dx,
                    (iz as f64 + 0.5) * dx,
                );
                let mut p = Particle::new(id, 1.0, pos, Vec3::zero());
                p.internal_energy = 1000.0;
                p.smoothing_length = 0.8 * dx;
                parts.push(p);
                if parts.len() >= n {
                    break 'outer;
                }
            }
        }
    }
    parts
}

fn make_black_holes(n_bh: usize, box_size: f64) -> Vec<BlackHole> {
    (0..n_bh)
        .map(|i| {
            let frac = (i as f64 + 0.5) / n_bh as f64;
            BlackHole {
                pos: Vec3::new(frac * box_size, frac * box_size, frac * box_size),
                mass: 1e8,
                accretion_rate: 1e-4,
            }
        })
        .collect()
}

fn bench_apply_agn_feedback(c: &mut Criterion) {
    let box_size = 50.0;
    let params = AgnParams {
        eps_feedback: 0.05,
        m_seed: 1e5,
        v_kick_agn: 0.0, // solo térmico, sin kick (más limpio para benchmark)
        r_influence: 5.0,
    };
    let dt = 0.01;

    let mut group = c.benchmark_group("apply_agn_feedback");

    // Barrido sobre número de partículas
    for &n_part in &[64_usize, 512, 4096, 32768] {
        let bhs = make_black_holes(1, box_size);
        group.throughput(Throughput::Elements(n_part as u64));
        group.bench_with_input(BenchmarkId::new("N_particles", n_part), &n_part, |b, &n| {
            let mut particles = make_gas_particles(n, box_size);
            b.iter(|| {
                apply_agn_feedback(&mut particles, &bhs, &params, dt);
            });
        });
    }

    // Barrido sobre número de agujeros negros (N_part fijo = 4096)
    let n_part_fixed = 4096;
    for &n_bh in &[1_usize, 4, 16] {
        let mut particles = make_gas_particles(n_part_fixed, box_size);
        let bhs = make_black_holes(n_bh, box_size);
        group.throughput(Throughput::Elements(n_bh as u64));
        group.bench_with_input(BenchmarkId::new("N_black_holes", n_bh), &n_bh, |b, _| {
            b.iter(|| {
                apply_agn_feedback(&mut particles, &bhs, &params, dt);
            });
        });
    }

    group.finish();
}

fn bench_bondi_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("bondi_accretion_rate");

    let bh_masses = [1e6_f64, 1e7, 1e8, 1e9];
    for &mass in &bh_masses {
        let bh = BlackHole {
            pos: Vec3::zero(),
            mass,
            accretion_rate: 0.0,
        };
        group.bench_with_input(
            BenchmarkId::new("M_bh", format!("{:.0e}", mass)),
            &mass,
            |b, _| {
                b.iter(|| bondi_accretion_rate(&bh, 1.0, 10.0));
            },
        );
    }

    group.finish();
}

fn bench_grow_black_holes(c: &mut Criterion) {
    let box_size = 50.0;
    let params = AgnParams {
        eps_feedback: 0.05,
        m_seed: 1e5,
        v_kick_agn: 0.0,
        r_influence: 5.0,
    };
    let dt = 0.01;

    let mut group = c.benchmark_group("grow_black_holes");

    for &n_part in &[64_usize, 512, 4096] {
        let particles = make_gas_particles(n_part, box_size);
        group.throughput(Throughput::Elements(n_part as u64));
        group.bench_with_input(BenchmarkId::new("N_particles", n_part), &n_part, |b, _| {
            let mut bhs = make_black_holes(1, box_size);
            b.iter(|| {
                grow_black_holes(&mut bhs, &particles, &params, dt);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_apply_agn_feedback,
    bench_bondi_rate,
    bench_grow_black_holes
);
criterion_main!(benches);
