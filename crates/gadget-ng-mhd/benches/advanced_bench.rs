//! Benchmarks avanzados MHD — Phase 143
//!
//! Mide el rendimiento de:
//! - `apply_turbulent_forcing` — N = 100, 500, 1000 partículas
//! - `apply_flux_freeze`       — N = 100, 500, 1000
//! - `advance_srmhd`           — N = 100 (~10% relativistas)
//! - `srmhd_conserved_to_primitive` — llamada unitaria, 1000 iteraciones

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use gadget_ng_core::{Particle, ParticleType, TurbulenceSection, Vec3};
use gadget_ng_mhd::{
    C_LIGHT, advance_srmhd, apply_flux_freeze, apply_turbulent_forcing, mean_gas_density,
    srmhd_conserved_to_primitive,
};

fn make_gas_particles(n: usize) -> Vec<Particle> {
    (0..n)
        .map(|i| {
            let fi = i as f64;
            let mut p = Particle::new_gas(
                i,
                1.0,
                Vec3::new(
                    fi * 0.01,
                    (fi * 0.013).sin() * 0.1,
                    (fi * 0.017).cos() * 0.1,
                ),
                Vec3::new(fi * 1e-3, 0.0, 0.0),
                1e10,
                0.2,
            );
            p.b_field = Vec3::new(1e-6 * (fi + 1.0), 0.0, 0.0);
            p
        })
        .collect()
}

fn turb_cfg() -> TurbulenceSection {
    TurbulenceSection {
        enabled: true,
        amplitude: 1e-3,
        correlation_time: 1.0,
        k_min: 1.0,
        k_max: 8.0,
        spectral_index: 5.0 / 3.0,
    }
}

// ── Turbulencia ────────────────────────────────────────────────────────────

fn bench_turbulent_forcing(c: &mut Criterion) {
    let cfg = turb_cfg();
    let mut group = c.benchmark_group("turbulent_forcing");
    for n in [100, 500, 1000] {
        group.bench_with_input(BenchmarkId::new("N", n), &n, |b, &n| {
            let mut particles = make_gas_particles(n);
            b.iter(|| {
                apply_turbulent_forcing(
                    black_box(&mut particles),
                    black_box(&cfg),
                    black_box(0.01),
                    black_box(42),
                );
            });
        });
    }
    group.finish();
}

// ── Flux-freeze ────────────────────────────────────────────────────────────

fn bench_flux_freeze(c: &mut Criterion) {
    let mut group = c.benchmark_group("flux_freeze");
    for n in [100, 500, 1000] {
        group.bench_with_input(BenchmarkId::new("N", n), &n, |b, &n| {
            let mut particles = make_gas_particles(n);
            let rho_ref = mean_gas_density(&particles);
            b.iter(|| {
                apply_flux_freeze(
                    black_box(&mut particles),
                    black_box(5.0 / 3.0),
                    black_box(100.0),
                    black_box(rho_ref),
                );
            });
        });
    }
    group.finish();
}

// ── SRMHD advance ──────────────────────────────────────────────────────────

fn bench_advance_srmhd(c: &mut Criterion) {
    let n = 100;
    let mut particles = make_gas_particles(n);
    // ~10% de partículas con velocidades relativistas
    for i in 0..(n / 10) {
        particles[i * 10].velocity = Vec3::new(0.5 * C_LIGHT, 0.0, 0.0);
    }
    c.bench_function("advance_srmhd_N100", |b| {
        b.iter(|| {
            advance_srmhd(
                black_box(&mut particles),
                black_box(1e-4),
                black_box(C_LIGHT),
                black_box(0.1),
            );
        });
    });
}

// ── srmhd_conserved_to_primitive unitaria ─────────────────────────────────

fn bench_conserved_to_primitive(c: &mut Criterion) {
    c.bench_function("srmhd_conserved_to_primitive_1000_iter", |b| {
        b.iter(|| {
            for seed in 0..1000u64 {
                let d = 1.0 + (seed as f64 * 0.001) % 10.0;
                let sx = (seed as f64 * 0.002) % 0.8 * C_LIGHT * d;
                let e_cons = d * C_LIGHT * C_LIGHT * 1.1 + 0.5 * sx * sx / d;
                let s_arr = [sx, 0.0, 0.0];
                let b_arr = [1e-6, 0.0, 0.0];
                let _ = srmhd_conserved_to_primitive(
                    black_box(d),
                    black_box(s_arr),
                    black_box(e_cons),
                    black_box(b_arr),
                    black_box(5.0 / 3.0),
                    black_box(C_LIGHT),
                );
            }
        });
    });
}

criterion_group!(
    benches,
    bench_turbulent_forcing,
    bench_flux_freeze,
    bench_advance_srmhd,
    bench_conserved_to_primitive
);
criterion_main!(benches);
